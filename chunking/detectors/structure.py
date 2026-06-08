from typing import List, Dict, Any

from chunking.domain import SECTION_HEADER_PATTERN


class StructureDetector:
    """Group page text into sections while preserving page provenance.

    Each returned section keeps `segments`: an ordered list of
    {"page": <1-based page>, "text": ...} so downstream block-building can attach
    an accurate page (or page range) to every chunk. `text` is the joined form,
    kept for any consumer that just wants the section body.
    """

    def segment(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sections = []
        current_section = "General"
        current_segments = []  # [{"page": int, "text": str}, ...]

        section_pattern = SECTION_HEADER_PATTERN

        def flush():
            if current_segments:
                sections.append({
                    "section": current_section,
                    "segments": list(current_segments),
                    "text": "\n".join(s["text"] for s in current_segments),
                })

        for page in pages:
            pnum = page.get("page_num", 0)
            buf = []
            for line in page["text"].split("\n"):
                match = section_pattern.match(line.strip())
                if match:
                    # close the current page's buffer into a segment, then the section
                    if buf:
                        current_segments.append({"page": pnum, "text": "\n".join(buf)})
                        buf = []
                    flush()
                    current_segments = []
                    current_section = match.group(1).title()
                else:
                    buf.append(line)
            if buf:
                current_segments.append({"page": pnum, "text": "\n".join(buf)})

        flush()
        return sections
