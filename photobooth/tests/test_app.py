import pytest
from pathlib import Path
from app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page_includes_flash_overlay(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b'id="flashOverlay"' in response.data
    assert b'id="captureBtn"' in response.data


def test_home_page_includes_shutter_sound_hook(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b'playShutterSound' in response.data


def test_gallery_page_renders_photos(client):
    response = client.get("/gallery")

    assert response.status_code == 200
    assert b'gallery' in response.data.lower()


def test_home_page_links_to_gallery(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b'/gallery' in response.data


def test_home_page_contains_fullscreen_fallback_logic(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b'fullscreenchange' in response.data
    assert b'fullscreen-mode' in response.data


def test_home_page_contains_live_clock(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b'id="liveClock"' in response.data


def test_home_page_contains_event_title_and_info(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b'event-title' in response.data
    assert b'info-pill' in response.data


def test_strip_page_renders_and_has_strip_controls(client):
    response = client.get("/strip")

    assert response.status_code == 200
    assert b'Create 4-Photo Strip' in response.data
    assert b'Generate Strip' in response.data


def test_print_strip_page_renders(client):
    response = client.get("/print_strip")

    assert response.status_code == 200
    assert b'Print Strip' in response.data
    assert b'window.print' in response.data


def test_gallery_finds_photos_even_when_working_directory_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    photo_dir = Path(app.root_path) / "static" / "photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    test_photo = photo_dir / "gallery_test.jpg"
    test_photo.write_bytes(b"fake-image")

    client = app.test_client()
    response = client.get("/gallery")

    assert response.status_code == 200
    assert b"gallery_test.jpg" in response.data
