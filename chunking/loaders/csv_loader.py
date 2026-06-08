import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CSVLoader:
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        df = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip', engine='python')

        if 'text' not in df.columns:
            raise ValueError(f"CSV must have a 'text' column. Found: {list(df.columns)}")

        id_col = 'file_name' if 'file_name' in df.columns else None
        records = []

        for i, row in df.iterrows():
            text = str(row['text']).strip()
            if not text or text == 'nan' or len(text) < 50:
                continue

            doc_id = str(row[id_col]).replace('.xml', '').replace('.txt', '') if id_col else f"row_{i}"
            records.append({
                "doc_id": doc_id,
                "text": text,
                "row_index": i
            })

        logger.info(f"Loaded {len(records)} valid rows from {file_path}")
        return records
