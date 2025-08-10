#!/usr/bin/env python3
"""
Seedr CLI Downloader for REST API v1
───────────────────────────────────
• Enumerates folders in the user’s Seedr account
• Lets the user pick a folder to download
• Downloads every file in that folder with a nice progress-bar
  (up to 3 files in parallel, each with its own bar)
• If a local file with the same name exists, compute its checksum and compare
  against the checksum returned by Seedr (field “hash”) and ask the user whether
  to re-download
• Credentials are taken from environment variables:
      SEEDR_EMAIL
      SEEDR_PASSWORD
-----------------------------------------------------------
Python ≥ 3.8 is recommended.
Third-party dependencies:
      pip install requests tqdm
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import queue
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEEDR_BASE_URL = "https://www.seedr.cc"
ENV_EMAIL = "SEEDR_EMAIL"
ENV_PASSWORD = "SEEDR_PASSWORD"

PARALLEL_DOWNLOADS = 4           # how many files to download concurrently
CHUNK_SIZE = 1024 * 256          # 256 KiB


class SeedrClient:
    """
    Thin wrapper around the Seedr REST v1 API using HTTP basic auth.
    """

    def __init__(self, email: str, password: str, session: Optional[requests.Session] = None):
        self.auth = HTTPBasicAuth(email, password)
        self.session = session or requests.Session()
        self.session.auth = self.auth

    # -------------------------- Helper HTTP methods ------------------------ #
    def _get(self, endpoint: str, **kwargs) -> requests.Response:
        url = f"{SEEDR_BASE_URL}/rest/{endpoint.lstrip('/')}"
        r = self.session.get(url, **kwargs)
        r.raise_for_status()
        return r

    # --------------------------- Folder / files ---------------------------- #
    def list_folder(self, folder_id: Optional[int] = None) -> Dict:
        endpoint = "folder" if folder_id is None else f"folder/{folder_id}"
        return self._get(endpoint).json()

    def download_file_stream(self, file_id: int) -> requests.Response:
        endpoint = f"file/{file_id}"
        return self._get(endpoint, stream=True, allow_redirects=True)

    # --------------------------- Convenience ------------------------------- #
    def get_user(self):
        return self._get("user").json()


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def sha1sum(path: str, buf: int = 128 * 1024) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def prompt_yes_no(question: str, default: str = "n") -> bool:
    choices = "Y/n" if default.lower() == "y" else "y/N"
    prompt = f"{question} [{choices}] "
    while True:
        ans = input(prompt).strip().lower()
        if not ans:
            ans = default.lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer y(es) or n(o).")


# --------------------------------------------------------------------------- #
# Download logic
# --------------------------------------------------------------------------- #
def should_download(local_path: str, remote_hash: Optional[str]) -> bool:
    """
    Decide whether a file needs downloading. May prompt the user.
    """
    if not os.path.exists(local_path):
        return True

    print(f"\n{os.path.basename(local_path)} already exists locally.")
    if remote_hash:
        local_hash = sha1sum(local_path)
        if local_hash == remote_hash:
            print("  Checksums match – skip download.")
            return False
        print("  Checksums differ!")
    else:
        print("  (Checksum info not available.)")

    return prompt_yes_no("  Re-download and overwrite?", default="n")


def download_file(
    client: SeedrClient,
    file_meta: Dict,
    dest_dir: str,
    position_pool: "queue.Queue[int]",
) -> None:
    """
    Download a single file, updating its individual tqdm bar.
    A free 'position' for tqdm is first acquired from position_pool and returned
    when finished, so at most PARALLEL_DOWNLOADS bars are visible.
    """
    file_id, name, remote_size, remote_hash = (
        file_meta["id"],
        file_meta["name"],
        int(file_meta.get("size", 0)),
        file_meta.get("hash"),
    )

    local_path = os.path.join(dest_dir, name)
    if not should_download(local_path, remote_hash):
        return

    # reserve a tqdm console line
    pos = position_pool.get()          # blocks until a position is free
    try:
        resp = client.download_file_stream(file_id)
        total = int(resp.headers.get("Content-Length", 0)) or remote_size

        tmp = local_path + ".part"
        with open(tmp, "wb") as out, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=name,
            leave=False,
            position=pos,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    out.write(chunk)
                    bar.update(len(chunk))

        os.replace(tmp, local_path)
        tqdm.write(f"Finished: {name}")
    except Exception as exc:
        tqdm.write(f"Error downloading {name}: {exc}")
        if os.path.exists(local_path + ".part"):
            os.remove(local_path + ".part")
        raise
    finally:
        position_pool.put(pos)         # release tqdm line


def download_all_files_in_folder(
    client: SeedrClient,
    folder_json: Dict,
    destination: str,
) -> None:
    os.makedirs(destination, exist_ok=True)

    files: List[Dict] = folder_json.get("files", [])
    if not files:
        print("Selected folder contains no files.")
        return

    print(f"Preparing to download {len(files)} file(s) to '{destination}'.\n")

    # pool with `PARALLEL_DOWNLOADS` unique tqdm line positions (0..N-1)
    pos_pool: "queue.Queue[int]" = queue.Queue()
    for i in range(PARALLEL_DOWNLOADS):
        pos_pool.put(i)

    # run downloads concurrently
    with cf.ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOADS) as exe:
        futures = [
            exe.submit(download_file, client, f, destination, pos_pool)
            for f in files
        ]
        # drive progress: wait for futures, propagate errors
        for fut in cf.as_completed(futures):
            exc = fut.exception()
            if exc:
                # shut down remaining downloads gracefully
                for other in futures:
                    other.cancel()
                raise exc

    print("\nAll done!")


# --------------------------------------------------------------------------- #
# Folder selection helpers
# --------------------------------------------------------------------------- #
def choose_folder_interactively(client: SeedrClient) -> Dict:
    root = client.list_folder()
    folders = root.get("folders", [])

    print("\nAvailable folders:")
    print(" 0) /  (root)")
    for idx, fld in enumerate(folders, start=1):
        print(f"{idx:2}) {fld['name']}  (id={fld['id']})")

    while True:
        choice = input("\nSelect folder number to download (default 0 = root): ").strip()
        if not choice:
            choice = "0"
        if not choice.isdigit():
            print("Please enter a valid numeric choice.")
            continue
        num = int(choice)
        if num == 0:
            return root
        if 1 <= num <= len(folders):
            return client.list_folder(folders[num - 1]["id"])
        print("Choice out of range.")


# --------------------------------------------------------------------------- #
# Boiler-plate & entry point
# --------------------------------------------------------------------------- #
def get_credentials_from_env() -> Tuple[str, str]:
    email, pwd = os.getenv(ENV_EMAIL), os.getenv(ENV_PASSWORD)
    if not email or not pwd:
        print(
            textwrap.dedent(
                f"""
                Error: Seedr credentials not found.

                Please set:
                    export {ENV_EMAIL}="you@example.com"
                    export {ENV_PASSWORD}="your_password"
                """
            ).strip(),
            file=sys.stderr,
        )
        sys.exit(1)
    return email, pwd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all files in a Seedr folder (parallel, checksummed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dest",
        default="seedr_downloads",
        help="Local directory to save the downloaded files",
    )
    parser.add_argument(
        "--folder-id",
        type=int,
        help="Seedr folder ID to download (skip interactive prompt)",
    )
    args = parser.parse_args()

    email, pwd = get_credentials_from_env()
    client = SeedrClient(email, pwd)

    folder_json = (
        client.list_folder(args.folder_id)
        if args.folder_id is not None
        else choose_folder_interactively(client)
    )

    try:
        download_all_files_in_folder(client, folder_json, args.dest)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
