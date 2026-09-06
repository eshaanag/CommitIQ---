import asyncio
import logging
import os
import re
import shutil
import stat
import urllib.parse
from pathlib import Path

import httpx

from backend.config import GITHUB_TOKEN, REPO_STORAGE_PATH

logger = logging.getLogger(__name__)


def _redact_secret(value: str) -> str:
    if GITHUB_TOKEN:
        value = value.replace(GITHUB_TOKEN, "[REDACTED_GITHUB_TOKEN]")
    return re.sub(r"https?://[^/@\s]+@github\.com", "https://[REDACTED]@github.com", value)


def _is_valid_github_name(value: str) -> bool:
    if value in {".", ".."}:
        return False
    return bool(re.fullmatch(r"[\w.-]+", value))


def sanitize_repo_url(url: str) -> str:
    """
    Sanitize and strip token credentials and user info from repository URLs.
    e.g. 'https://token@github.com/owner/repo' -> 'https://github.com/owner/repo'
         'https://user:token@github.com/owner/repo' -> 'https://github.com/owner/repo'
         'http://token@github.com/owner/repo' -> 'http://github.com/owner/repo'
         'token@github.com/owner/repo' -> 'github.com/owner/repo'
    """
    if not url:
        return ""
    cleaned = re.sub(r"^(https?://)[^/@\s]+@", r"\1", url.strip())
    cleaned = re.sub(r"^[^/@\s]+@(github\.com[/:]|www\.github\.com[/:])", r"\1", cleaned)
    return cleaned


def parse_github_url(url: str) -> tuple[str, str]:
    """Parse GitHub URL or shorthand to ('owner', 'repo')."""
    s = sanitize_repo_url(url).strip()
    if not s:
        raise ValueError("Cannot parse empty GitHub URL.")

    # Strip trailing slashes and .git
    while s.endswith("/") or s.endswith(".git"):
        if s.endswith("/"):
            s = s[:-1]
        elif s.endswith(".git"):
            s = s[:-4]

    # Normalize to full URL for urlsplit
    has_scheme = s.startswith("http://") or s.startswith("https://")
    url_to_parse = s if has_scheme else f"https://{s}"

    parsed = urllib.parse.urlsplit(url_to_parse)
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parsed.path.strip("/")
    parts = [p for p in path.split("/") if p]

    if netloc == "github.com":
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Cannot parse GitHub URL: {_redact_secret(url)}. Expected format 'owner/repo' or 'github.com/owner/repo'."
            )
        owner, repo = parts[0], parts[1]
    elif not has_scheme and not netloc and len(parts) == 2:
        # Shorthand 'owner/repo'
        owner, repo = parts[0], parts[1]
    elif not has_scheme and len(parts) == 1 and "/" in s:
        # Shorthand where netloc caught the owner (e.g. owner/repo without scheme)
        owner, repo = netloc, parts[0]
    else:
        raise ValueError(
            f"Cannot parse GitHub URL: {_redact_secret(url)}. Expected a github.com repository URL."
        )

    if not _is_valid_github_name(owner) or not _is_valid_github_name(repo):
        raise ValueError(f"Invalid owner or repository name in URL: {_redact_secret(url)}")

    return owner, repo


def make_repo_slug(owner: str, repo: str) -> str:
    slug = f"{owner}-{repo}".lower()
    slug = re.sub(r"[^a-z0-9\-]", "-", slug)
    return slug


def get_storage_usage_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError as exc:
            logger.debug("Could not read file size for %s: %s", p, exc)
    return total / (1024 * 1024)


def get_clone_path(repo_id: int) -> Path:
    return REPO_STORAGE_PATH / str(repo_id)


async def clone_repo(
    repo_url: str,
    repo_id: int,
    max_commits: int = 150,
    branch: str | None = None,
) -> Path:
    """Shallow clone to local disk. Returns clone path."""
    target = get_clone_path(repo_id)
    if target.exists():
        if not cleanup_repo(repo_id):
            raise RuntimeError(f"Could not clean existing clone directory for repo_id={repo_id}")

    from backend.config import MAX_REPO_STORAGE_MB

    current_usage = get_storage_usage_mb(REPO_STORAGE_PATH)
    if current_usage >= MAX_REPO_STORAGE_MB:
        raise ValueError(
            f"Storage quota exceeded. Current: {current_usage:.1f} MB, Limit: {MAX_REPO_STORAGE_MB} MB"
        )

    target.mkdir(parents=True, exist_ok=True)

    # Use git clone via HTTPS — never uses GitHub REST API, no rate limits
    if GITHUB_TOKEN:
        auth_url = repo_url.replace("https://github.com/", f"https://{GITHUB_TOKEN}@github.com/")
    else:
        auth_url = repo_url

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""}

    clone_cmd = [
        "git",
        "clone",
        "--depth",
        str(max_commits),
        "--single-branch",
    ]

    if branch:
        clone_cmd.extend(["--branch", branch])

    clone_cmd.extend(
        [
            auth_url,
            str(target),
        ]
    )

    process = await asyncio.create_subprocess_exec(
        *clone_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=300,
        )
    except asyncio.TimeoutError:
        process.kill()
        cleanup_repo(repo_id)
        raise RuntimeError(f"git clone timed out for repo_id={repo_id}")

    if process.returncode != 0:
        cleanup_repo(repo_id)
        stderr_text = stderr.decode("utf-8", errors="replace")
        if (
            "not found" in stderr_text.lower()
            or "could not read" in stderr_text.lower()
            or "authentication failed" in stderr_text.lower()
        ):
            raise RuntimeError(f"Repository not found on GitHub or is private: {repo_url}")
        raise RuntimeError(f"git clone failed: {_redact_secret(stderr_text)[:500]}")

    return target


async def count_available_commits(repo_path: Path) -> int:
    process = await asyncio.create_subprocess_exec(
        "git",
        "rev-list",
        "--count",
        "HEAD",
        cwd=str(repo_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except asyncio.TimeoutError:
        process.kill()
        return 0

    if process.returncode != 0:
        return 0
    try:
        return int(stdout.decode("utf-8", errors="replace").strip())
    except ValueError:
        return 0


def _remove_readonly(func, path, _excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError as exc:
        logger.debug("Could not change file permissions on %s: %s", path, exc)


def cleanup_repo(repo_id: int) -> bool:
    """Delete cloned repo after ingestion to reclaim disk space."""
    target = get_clone_path(repo_id)
    if not target.exists():
        return True

    try:
        shutil.rmtree(target, onerror=_remove_readonly)
        return True
    except OSError as exc:
        logger.warning(
            "Could not clean cloned repo directory",
            extra={"repo_id": repo_id, "path": str(target), "error": str(exc)},
        )
        return False


async def fetch_github_metadata(owner: str, repo: str) -> dict:
    """Optional metadata fetch — cosmetic only, skipped on rate limit."""
    if not re.fullmatch(r"^[a-zA-Z0-9_.-]{1,100}$", owner) or not re.fullmatch(
        r"^[a-zA-Z0-9_.-]{1,100}$", repo
    ):
        return {
            "github_stars": None,
            "github_language": None,
            "github_description": None,
            "default_branch": "main",
        }

    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    safe_owner = urllib.parse.quote(owner, safe="")
    safe_repo = urllib.parse.quote(repo, safe="")

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"https://api.github.com/repos/{safe_owner}/{safe_repo}", headers=headers
            )
            if r.status_code == 200:
                data = r.json()
                return {
                    "github_stars": data.get("stargazers_count"),
                    "github_language": data.get("language"),
                    "github_description": data.get("description", "")[:300],
                    "default_branch": data.get("default_branch", "main"),
                }
    except Exception as exc:
        logger.debug("Failed to fetch GitHub metadata: %s", exc)
    return {
        "github_stars": None,
        "github_language": None,
        "github_description": None,
        "default_branch": "main",
    }


GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
GRAPHQL_PR_QUERY = """
query GetPullRequests($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(first: 100, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        state
        createdAt
        mergedAt
        closedAt
        author {
          login
        }
      }
    }
  }
}
"""


async def fetch_github_pull_requests_graphql(
    owner: str, repo: str, limit: int = 500
) -> list[dict] | None:
    """Fetch pull requests via GitHub GraphQL API to reduce rate limit consumption."""
    try:
        from dateutil.parser import parse as parse_date
    except ImportError:

        def parse_date(s: str):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))

    headers = {"Accept": "application/json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"bearer {GITHUB_TOKEN}"
    else:
        # GraphQL queries require authentication on GitHub API; return None to fall back to REST
        return None

    prs = []
    cursor = None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            while True:
                variables = {"owner": owner, "repo": repo, "cursor": cursor}
                response = await client.post(
                    GITHUB_GRAPHQL_URL,
                    json={"query": GRAPHQL_PR_QUERY, "variables": variables},
                    headers=headers,
                )
                if response.status_code != 200:
                    logger.warning(
                        f"GraphQL PR query returned status {response.status_code} for {owner}/{repo}"
                    )
                    return None

                data = response.json()
                if "errors" in data or "data" not in data or not data["data"].get("repository"):
                    logger.warning(
                        f"GraphQL PR query errors or empty repository for {owner}/{repo}"
                    )
                    return None

                pr_data = data["data"]["repository"]["pullRequests"]
                nodes = pr_data.get("nodes") or []
                page_info = pr_data.get("pageInfo") or {}

                for item in nodes:
                    created_at = item.get("createdAt")
                    merged_at = item.get("mergedAt")
                    closed_at = item.get("closedAt")
                    author_obj = item.get("author") or {}
                    author_login = (
                        author_obj.get("login", "unknown")
                        if isinstance(author_obj, dict)
                        else "unknown"
                    )

                    state_raw = (item.get("state") or "unknown").lower()

                    prs.append(
                        {
                            "pr_number": item.get("number"),
                            "title": item.get("title", "")[:255],
                            "state": state_raw,
                            "author": author_login,
                            "created_at": parse_date(created_at) if created_at else None,
                            "merged_at": parse_date(merged_at) if merged_at else None,
                            "closed_at": parse_date(closed_at) if closed_at else None,
                        }
                    )

                if len(prs) >= limit:
                    prs = prs[:limit]
                    break

                if not page_info.get("hasNextPage") or not page_info.get("endCursor"):
                    break

                cursor = page_info.get("endCursor")
        return prs
    except Exception as e:
        logger.warning(f"Error fetching PRs via GraphQL for {owner}/{repo}: {e}")
        return None


async def fetch_github_pull_requests_rest(owner: str, repo: str, limit: int = 500) -> list[dict]:
    """Fallback fetch for pull requests using REST API."""
    try:
        from dateutil.parser import parse as parse_date
    except ImportError:

        def parse_date(s: str):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))

    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    safe_owner = urllib.parse.quote(owner, safe="")
    safe_repo = urllib.parse.quote(repo, safe="")

    prs = []
    page = 1
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            while True:
                r = await client.get(
                    f"https://api.github.com/repos/{safe_owner}/{safe_repo}/pulls?state=all&per_page=100&page={page}",
                    headers=headers,
                )
                if r.status_code != 200:
                    break
                data = r.json()
                if not data:
                    break
                for item in data:
                    created_at = item.get("created_at")
                    merged_at = item.get("merged_at")
                    closed_at = item.get("closed_at")

                    prs.append(
                        {
                            "pr_number": item.get("number"),
                            "title": item.get("title", "")[:255],
                            "state": item.get("state", "unknown"),
                            "author": (
                                item.get("user", {}).get("login", "unknown")
                                if item.get("user")
                                else "unknown"
                            ),
                            "created_at": parse_date(created_at) if created_at else None,
                            "merged_at": parse_date(merged_at) if merged_at else None,
                            "closed_at": parse_date(closed_at) if closed_at else None,
                        }
                    )

                if len(prs) >= limit:
                    prs = prs[:limit]
                    break
                page += 1
    except Exception as e:
        logger.warning(f"Error fetching PRs via REST for {owner}/{repo}: {e}")
    return prs


async def fetch_github_pull_requests(owner: str, repo: str, limit: int = 500) -> list[dict]:
    """Fetch pull requests using GraphQL API with automatic fallback to REST API."""
    if GITHUB_TOKEN:
        graphql_prs = await fetch_github_pull_requests_graphql(owner, repo, limit)
        if graphql_prs is not None:
            return graphql_prs

    return await fetch_github_pull_requests_rest(owner, repo, limit)
