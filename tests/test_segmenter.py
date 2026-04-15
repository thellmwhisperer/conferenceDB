from ingest import TranscriptEntry, build_segments


def test_build_segments_prefers_sentence_boundary_once_target_window_is_met():
    entries = [
        TranscriptEntry(1, 0.0, 10.0, "Welcome everyone to Umprompted"),
        TranscriptEntry(2, 10.0, 21.0, "Today we talk about prompt injection."),
        TranscriptEntry(3, 21.0, 30.0, "Attackers chain model outputs into tools"),
        TranscriptEntry(4, 30.0, 39.0, "and then pivot into the rest of the environment."),
    ]

    segments = build_segments("day2_12", entries)

    assert len(segments) == 2
    assert segments[0].start_time == 0.0
    assert segments[0].end_time == 21.0
    assert segments[0].text.endswith("prompt injection.")
    assert segments[1].start_time == 21.0
    assert segments[1].end_time == 39.0


def test_build_segments_uses_pause_as_boundary():
    entries = [
        TranscriptEntry(1, 0.0, 12.0, "Agents are already in production"),
        TranscriptEntry(2, 12.0, 24.0, "and we can measure where they fail."),
        TranscriptEntry(3, 26.5, 34.0, "A separate talk focused on red teaming."),
    ]

    segments = build_segments("day1_01", entries)

    assert len(segments) == 2
    assert segments[0].end_time == 24.0
    assert segments[1].start_time == 26.5

