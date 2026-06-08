import re
import unicodedata

# OCR artifacts: Private Use Area glyphs (special-font ligatures that render as
# garbage, e.g. "<min" for "<1 min") and stray C0/C1 control chars (keep \t \n \r).
_PUA = re.compile("[-\U000f0000-\U0010fffd]")
_CTRL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Unicode replacement char (U+FFFD): an undecodable glyph, e.g. a bullet PyMuPDF
# couldn't map. Drop it along with any adjacent tab so list markers don't leave gaps.
_REPLACEMENT = re.compile("�\t?")


class TextCleaner:
    def normalize(self, text: str) -> str:
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Strip OCR garbage glyphs and stray control characters.
        text = _PUA.sub("", text)
        text = _CTRL.sub("", text)
        text = _REPLACEMENT.sub("", text)

        # Remove whitespace noise
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove typical headers/footers and noise
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line_s = line.strip()

            # Remove isolated tokens (S1, t1, 1, a)
            if re.fullmatch(r'[A-Za-z0-9]{1,2}', line_s):
                continue

            # Remove captions / figures / references
            if re.match(r'^(figure|fig\.|table|video|references?)\b', line_s, re.IGNORECASE):
                continue

            if re.match(r'^Page \d+$', line_s) or re.match(r'^\d+$', line_s):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)
