from pyampp.gxbox.gxampp import _split_process_output_text


def test_split_process_output_text_handles_partial_line_boundaries():
    complete_lines, partial = _split_process_output_text("partial", " line\nnext")

    assert complete_lines == ["partial line"]
    assert partial == "next"


def test_split_process_output_text_normalizes_crlf_and_flushes_complete_tail():
    complete_lines, partial = _split_process_output_text("", "alpha\r\nbeta\rgamma\n")

    assert complete_lines == ["alpha", "beta", "gamma"]
    assert partial == ""