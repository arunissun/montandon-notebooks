from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from urllib.parse import quote


EDIT_LINK_NOTEBOOK_RE = re.compile(
    r'href="https://github\.com/[^"#?]+/edit/[^"#?]+/book/notebooks/([^"#?]+\.ipynb)"'
)

EDIT_LINK_ANCHOR_RE = re.compile(
    r'(<a[^>]*class="myst-fm-edit-link[^"]*"[^>]*>.*?</a>)',
    re.DOTALL,
)


def make_binder_href(binder_url: str, binder_repo: str, binder_ref: str, notebook_rel: str) -> str:
    notebook_path = f"book/notebooks/{notebook_rel}"
    encoded_path = quote(notebook_path, safe="/")
    base = binder_url.rstrip("/")
    return f"{base}/v2/gh/{binder_repo}/{binder_ref}?urlpath=lab/tree/{encoded_path}"


def make_button_html(binder_href: str) -> str:
    safe_href = html.escape(binder_href, quote=True)
    return (
        f'<a data-binder-inject="true" href="{safe_href}" '
        'title="Open this notebook in Binder" target="_blank" rel="noopener noreferrer" '
        'class="myst-fm-binder-link text-inherit hover:text-inherit ml-1">'
        '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
        'stroke-width="1.5" stroke="currentColor" aria-hidden="true" data-slot="icon" '
        'width="1.25rem" height="1.25rem" '
        'class="myst-fm-binder-icon inline-block mr-1 opacity-60 hover:opacity-100">'
        '<title>Open in Binder</title>'
        '<path stroke-linecap="round" stroke-linejoin="round" '
        'd="M9 3.75h6m-6 0 1.5 3h3L15 3.75M9 3.75 5.25 9h13.5L15 3.75M6 9.75h12l-1.2 9.3a2.25 2.25 0 0 1-2.23 1.95H9.43A2.25 2.25 0 0 1 7.2 19.05L6 9.75Z"/>'
        '</svg></a>'
    )


def inject_button(content: str, binder_url: str, binder_repo: str, binder_ref: str) -> tuple[str, bool]:
    if 'data-binder-inject="true"' in content:
        return content, False

    notebook_match = EDIT_LINK_NOTEBOOK_RE.search(content)
    if not notebook_match:
        return content, False

    notebook_rel = notebook_match.group(1)
    binder_href = make_binder_href(binder_url, binder_repo, binder_ref, notebook_rel)
    button_html = make_button_html(binder_href)

    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{button_html}"

    updated, count = EDIT_LINK_ANCHOR_RE.subn(_repl, content, count=1)
    if count == 0:
        return content, False
    return updated, True


def iter_index_files(build_dir: Path) -> list[Path]:
    return sorted(build_dir.glob("**/index.html"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject direct Binder buttons into built notebook pages")
    parser.add_argument("--build-dir", required=True, help="Path to built HTML directory, e.g. book/_build/html")
    parser.add_argument("--binder-url", required=True, help="BinderHub base URL, e.g. https://mybinder.org")
    parser.add_argument("--binder-repo", required=True, help="GitHub repo in owner/name format")
    parser.add_argument("--binder-ref", default="main", help="Git ref/branch for Binder")
    parser.add_argument("--dry-run", action="store_true", help="Report files that would change without writing")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    if not build_dir.exists():
        raise SystemExit(f"Build directory not found: {build_dir}")

    updated_files = 0
    scanned_files = 0

    for html_file in iter_index_files(build_dir):
        scanned_files += 1
        original = html_file.read_text(encoding="utf-8")
        updated, changed = inject_button(
            original,
            binder_url=args.binder_url,
            binder_repo=args.binder_repo,
            binder_ref=args.binder_ref,
        )

        if not changed:
            continue

        updated_files += 1
        if not args.dry_run:
            html_file.write_text(updated, encoding="utf-8")

    mode = "DRY RUN" if args.dry_run else "WRITE"
    print(f"[{mode}] scanned={scanned_files} updated={updated_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
