"""Fetch the authors' public weights, with content hashes and no account needed."""
import hashlib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
import zipfile

import requests

TRACKNET_SHA256 = "df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a"
ARCHIVE_SHA256 = "8de68d34ea2457e368e22947f1309143d60a6a7f9ddccd31cd526d4c4571bcc4"
POSE_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"


def fetch_pose(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "yolo26n-pose.pt"
    if target.exists() and sha256(target) == POSE_SHA256:
        return target
    temporary = target.with_suffix(".part")
    url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    if sha256(temporary) != POSE_SHA256:
        temporary.unlink()
        raise RuntimeError("Pose checkpoint checksum mismatch")
    temporary.replace(target)
    return target


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.action = attrs.get("action")
        if tag == "input" and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value", "")


def fetch_tracknet(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "TrackNet_best.pt"
    if target.exists() and sha256(target) == TRACKNET_SHA256:
        return target
    archive = directory / "TrackNetV3_ckpts.zip"
    if not archive.exists() or sha256(archive) != ARCHIVE_SHA256:
        with requests.Session() as session:
            response = session.get("https://drive.google.com/uc", params={
                "export": "download", "id": "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA",
            }, stream=True, timeout=120)
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", ""):
                form = DownloadForm()
                form.feed(response.text)
                if urlparse(form.action or "").hostname != "drive.usercontent.google.com":
                    raise RuntimeError("Google Drive did not provide its public download form")
                response.close()
                response = session.get(form.action, params=form.fields, stream=True, timeout=120)
                response.raise_for_status()
            temporary = archive.with_suffix(".part")
            with temporary.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
            response.close()
        if sha256(temporary) != ARCHIVE_SHA256:
            temporary.unlink()
            raise RuntimeError("TrackNet archive checksum mismatch")
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        member, = [name for name in bundle.namelist() if Path(name).name == target.name]
        temporary = target.with_suffix(".part")
        with bundle.open(member) as source, temporary.open("wb") as destination:
            import shutil
            shutil.copyfileobj(source, destination)
    if sha256(temporary) != TRACKNET_SHA256:
        temporary.unlink()
        raise RuntimeError("TrackNet checkpoint checksum mismatch")
    temporary.replace(target)
    return target
