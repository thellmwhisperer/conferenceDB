import json

from ingest import load_manifest


def test_load_manifest_uses_manifest_metadata_and_defaults(tmp_path):
    manifest_path = tmp_path / "demo.json"
    manifest_path.write_text(
        json.dumps(
            {
                "conference": {
                    "id": "demo-conf",
                    "name": "Demo Conference",
                    "slug": "demo",
                    "video_dir": "videos",
                    "output_dir": "output"
                },
                "sessions": [
                    {
                        "id": "session_01",
                        "title": "Birds-of-a-Feather: Secure Agents",
                        "speakers": ["Chris Cochran"]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    conference, sessions = load_manifest(manifest_path, tmp_path / "fallback-videos", tmp_path / "fallback-output")

    assert conference.conference_id == "demo-conf"
    assert conference.slug == "demo"
    assert conference.video_dir == (tmp_path / "videos").resolve()
    assert conference.output_dir == (tmp_path / "output").resolve()
    assert sessions[0].conference_id == "demo-conf"
    assert sessions[0].session_type == "birds_of_a_feather"
    assert sessions[0].speakers[0].display_name == "Chris Cochran"
