"""Legacy entry point retained for users of the old local project.

Document ingestion now happens from the Gradio sidebar in ``webUI.py``. The
old Chroma/OneAPI implementation was removed because it was unrelated to the
public session-only demo and contained an obsolete embedded credential.
"""


def main() -> None:
    print("请运行 python webUI.py，并在网页侧栏上传 PDF 或 TXT。")


if __name__ == "__main__":
    main()
