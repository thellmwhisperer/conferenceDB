import json

from classify_turns import (
    SegmentMeasurement,
    classify_segments,
    compute_baseline_lufs,
    load_measurements,
    write_jsonl,
)


def test_compute_baseline_lufs_uses_early_session_window():
    segments = [
        SegmentMeasurement(index=1, start_time=0.0, end_time=8.0, text="Intro", lufs=-19.5),
        SegmentMeasurement(index=2, start_time=8.0, end_time=16.0, text="Context", lufs=-20.0),
        SegmentMeasurement(index=3, start_time=16.0, end_time=24.0, text="Details", lufs=-20.5),
        SegmentMeasurement(index=4, start_time=900.0, end_time=910.0, text="Question", lufs=-29.0),
    ]

    baseline = compute_baseline_lufs(segments, baseline_window_secs=60.0)

    assert baseline == -20.0


def test_classify_segments_labels_audience_uncertain_and_speaker():
    segments = [
        SegmentMeasurement(index=1, start_time=0.0, end_time=8.0, text="Welcome everyone.", lufs=-19.5),
        SegmentMeasurement(index=2, start_time=8.0, end_time=16.0, text="Today we talk about secure agents.", lufs=-19.8),
        SegmentMeasurement(index=3, start_time=16.0, end_time=24.0, text="This matters in production.", lufs=-20.1),
        SegmentMeasurement(index=4, start_time=24.0, end_time=32.0, text="What about prompt injection?", lufs=-27.8),
        SegmentMeasurement(index=5, start_time=32.0, end_time=40.0, text="Great question. Let's cover that.", lufs=-20.2),
        SegmentMeasurement(index=6, start_time=40.0, end_time=47.0, text="Can you say more about policy enforcement?", lufs=-23.6),
    ]

    classified = classify_segments(segments, baseline_window_secs=60.0, neighbor_radius=1)

    assert classified[0].label == "speaker"
    assert classified[3].label == "audience"
    assert classified[5].label == "uncertain"
    assert classified[3].delta_from_baseline >= 7.0


def test_question_like_text_with_normal_loudness_stays_speaker():
    segments = [
        SegmentMeasurement(index=1, start_time=0.0, end_time=9.0, text="Today we are covering policy.", lufs=-19.7),
        SegmentMeasurement(index=2, start_time=9.0, end_time=15.0, text="What about prompt injection in build pipelines?", lufs=-20.3),
        SegmentMeasurement(index=3, start_time=15.0, end_time=26.0, text="Let's walk through the controls in order.", lufs=-19.8),
    ]

    classified = classify_segments(segments, baseline_window_secs=60.0, neighbor_radius=1)

    assert classified[1].question_like is True
    assert classified[1].label == "speaker"


def test_low_loudness_question_between_long_turns_is_audience():
    segments = [
        SegmentMeasurement(
            index=1,
            start_time=0.0,
            end_time=14.0,
            text="We built the system so every orchestration step can be evaluated against policy in a deterministic way.",
            lufs=-19.8,
        ),
        SegmentMeasurement(index=2, start_time=14.0, end_time=19.0, text="Can you say more about human approval?", lufs=-26.8),
        SegmentMeasurement(
            index=3,
            start_time=19.0,
            end_time=33.0,
            text="Yes, for mutation and data sharing we require confirmation and record that decision in the workflow.",
            lufs=-20.1,
        ),
    ]

    classified = classify_segments(segments, baseline_window_secs=60.0, neighbor_radius=1)

    assert classified[1].question_like is True
    assert classified[1].label == "audience"


def test_short_question_between_long_turns_with_normal_loudness_is_uncertain():
    segments = [
        SegmentMeasurement(
            index=1,
            start_time=0.0,
            end_time=16.0,
            text="We are going to enforce these controls at every orchestration step because the agent can mutate state and exfiltrate data.",
            lufs=-19.9,
        ),
        SegmentMeasurement(index=2, start_time=16.0, end_time=20.0, text="Dan, what's up?", lufs=-20.7),
        SegmentMeasurement(
            index=3,
            start_time=20.0,
            end_time=36.0,
            text="Then after that we will walk through the approval flow and show where the policy engine blocks the action.",
            lufs=-20.0,
        ),
    ]

    classified = classify_segments(segments, baseline_window_secs=60.0, neighbor_radius=1)

    assert classified[1].question_like is True
    assert classified[1].label == "uncertain"


def test_load_measurements_reads_json_list(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 5.0, "lufs": -19.4, "text": "Hello"},
                {"start": 5.0, "end": 10.0, "lufs": -27.0, "text": "Question?"},
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_measurements(path)

    assert len(loaded) == 2
    assert loaded[0].index == 1
    assert loaded[1].lufs == -27.0


def test_write_jsonl_emits_qa_friendly_rows(tmp_path):
    segments = [
        SegmentMeasurement(index=1, start_time=0.0, end_time=8.0, text="Welcome everyone.", lufs=-19.5),
        SegmentMeasurement(index=2, start_time=8.0, end_time=16.0, text="What about prompt injection?", lufs=-27.8),
    ]
    classified = classify_segments(segments, baseline_window_secs=60.0, neighbor_radius=1)

    output_path = tmp_path / "roles.jsonl"
    write_jsonl(output_path, "day2_07", classified)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_id"] == "day2_07"
    assert rows[1]["label"] == "audience"
    assert "delta_from_baseline" in rows[1]
    assert isinstance(rows[1]["reasons"], list)
