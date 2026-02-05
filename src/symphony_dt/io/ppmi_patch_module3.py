"""Module 3 patch note.

In the full repo, Module 3 updated symphony_dt/io/ppmi.py to:
  - normalize columns via symphony_dt.schemas.normalize.normalize_columns
  - validate with contracts

This patch zip provides the schemas + notebook ingest. Apply the ppmi.py change by importing normalize_columns
and using validate_table where appropriate in your repo.
"""
