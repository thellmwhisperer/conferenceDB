from ingest import parse_srt_content


def test_parse_srt_content_reads_standard_blocks():
    content = """1
00:00:00,000 --> 00:00:07,000
Welcome everyone to the opening session of Umprompted.

2
00:00:07,000 --> 00:00:14,000
Today we're going to explore
how offensive AI is reshaping the threat landscape.
"""

    entries = parse_srt_content(content)

    assert len(entries) == 2
    assert entries[0].index == 1
    assert entries[0].start_time == 0.0
    assert entries[0].end_time == 7.0
    assert entries[0].text == "Welcome everyone to the opening session of Umprompted."
    assert entries[1].text == "Today we're going to explore how offensive AI is reshaping the threat landscape."


def test_parse_srt_content_tolerates_missing_index():
    content = """00:00:15,500 --> 00:00:20,000
Prompt injection is not theoretical anymore.
"""

    entries = parse_srt_content(content)

    assert len(entries) == 1
    assert entries[0].index == 1
    assert entries[0].start_time == 15.5
    assert entries[0].end_time == 20.0

