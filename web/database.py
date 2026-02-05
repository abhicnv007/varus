"""
SQLite database for patient annotations.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    view_type TEXT NOT NULL CHECK (view_type IN ('ap', 'lateral')),
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
                    UNIQUE (patient_id, view_type)
                );

                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL UNIQUE,
                    landmarks TEXT NOT NULL,
                    annotator TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_images_patient ON images(patient_id);
                CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations(image_id);
            """)

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Patient operations
    def create_patient(self, name: str, notes: Optional[str] = None) -> int:
        """Create a new patient. Returns patient ID."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO patients (name, notes, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (name, notes, now, now)
            )
            return cursor.lastrowid

    def get_patient(self, patient_id: int) -> Optional[dict]:
        """Get patient by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_patient_by_name(self, name: str) -> Optional[dict]:
        """Get patient by name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE name = ?", (name,)
            ).fetchone()
            return dict(row) if row else None

    def list_patients(self) -> list[dict]:
        """List all patients with their annotation status."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT
                    p.id,
                    p.name,
                    p.notes,
                    p.created_at,
                    p.updated_at,
                    GROUP_CONCAT(DISTINCT i.view_type) as views,
                    GROUP_CONCAT(DISTINCT CASE WHEN a.id IS NOT NULL THEN i.view_type END) as annotated_views
                FROM patients p
                LEFT JOIN images i ON i.patient_id = p.id
                LEFT JOIN annotations a ON a.image_id = i.id
                GROUP BY p.id
                ORDER BY p.created_at DESC
            """).fetchall()

            result = []
            for row in rows:
                d = dict(row)
                views = set(d['views'].split(',')) if d['views'] else set()
                annotated = set(d['annotated_views'].split(',')) if d['annotated_views'] else set()
                d['has_ap'] = 'ap' in views
                d['has_lateral'] = 'lateral' in views
                d['ap_annotated'] = 'ap' in annotated
                d['lateral_annotated'] = 'lateral' in annotated
                d['complete'] = d['ap_annotated'] and d['lateral_annotated']
                del d['views']
                del d['annotated_views']
                result.append(d)
            return result

    def delete_patient(self, patient_id: int) -> bool:
        """Delete a patient and all associated data."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
            return cursor.rowcount > 0

    # Image operations
    def add_image(self, patient_id: int, view_type: str, filename: str) -> int:
        """Add an image for a patient. Returns image ID."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            # Check if image already exists for this patient/view
            existing = conn.execute(
                "SELECT id, filename FROM images WHERE patient_id = ? AND view_type = ?",
                (patient_id, view_type)
            ).fetchone()

            if existing:
                # Update existing image
                conn.execute(
                    "UPDATE images SET filename = ? WHERE id = ?",
                    (filename, existing['id'])
                )
                # Delete old annotation if exists
                conn.execute("DELETE FROM annotations WHERE image_id = ?", (existing['id'],))
                return existing['id']

            cursor = conn.execute(
                "INSERT INTO images (patient_id, view_type, filename, created_at) VALUES (?, ?, ?, ?)",
                (patient_id, view_type, filename, now)
            )
            return cursor.lastrowid

    def get_image(self, image_id: int) -> Optional[dict]:
        """Get image by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_patient_images(self, patient_id: int) -> list[dict]:
        """Get all images for a patient."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT i.*,
                    CASE WHEN a.id IS NOT NULL THEN 1 ELSE 0 END as has_annotation
                FROM images i
                LEFT JOIN annotations a ON a.image_id = i.id
                WHERE i.patient_id = ?""",
                (patient_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    # Annotation operations
    def save_annotation(self, image_id: int, landmarks: dict, annotator: Optional[str] = None) -> int:
        """Save or update annotation for an image. Returns annotation ID."""
        now = datetime.utcnow().isoformat()
        landmarks_json = json.dumps(landmarks)

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, created_at FROM annotations WHERE image_id = ?",
                (image_id,)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE annotations SET landmarks = ?, annotator = ?, updated_at = ? WHERE id = ?",
                    (landmarks_json, annotator, now, existing['id'])
                )
                return existing['id']

            cursor = conn.execute(
                "INSERT INTO annotations (image_id, landmarks, annotator, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (image_id, landmarks_json, annotator, now, now)
            )
            return cursor.lastrowid

    def get_annotation(self, image_id: int) -> Optional[dict]:
        """Get annotation for an image."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM annotations WHERE image_id = ?", (image_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d['landmarks'] = json.loads(d['landmarks'])
                return d
            return None

    def get_patient_annotation(self, patient_id: int, view_type: str) -> Optional[dict]:
        """Get annotation for a patient's specific view."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT a.*, i.filename, i.view_type
                FROM annotations a
                JOIN images i ON i.id = a.image_id
                WHERE i.patient_id = ? AND i.view_type = ?""",
                (patient_id, view_type)
            ).fetchone()
            if row:
                d = dict(row)
                d['landmarks'] = json.loads(d['landmarks'])
                return d
            return None

    # Stats
    def get_stats(self) -> dict:
        """Get annotation statistics."""
        with self._connect() as conn:
            total_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

            stats = conn.execute("""
                SELECT
                    COUNT(DISTINCT CASE WHEN i.view_type = 'ap' THEN i.patient_id END) as has_ap,
                    COUNT(DISTINCT CASE WHEN i.view_type = 'lateral' THEN i.patient_id END) as has_lateral,
                    COUNT(DISTINCT CASE WHEN i.view_type = 'ap' AND a.id IS NOT NULL THEN i.patient_id END) as ap_annotated,
                    COUNT(DISTINCT CASE WHEN i.view_type = 'lateral' AND a.id IS NOT NULL THEN i.patient_id END) as lateral_annotated
                FROM images i
                LEFT JOIN annotations a ON a.image_id = i.id
            """).fetchone()

            # Count fully complete patients
            complete = conn.execute("""
                SELECT COUNT(*) FROM (
                    SELECT p.id
                    FROM patients p
                    JOIN images i_ap ON i_ap.patient_id = p.id AND i_ap.view_type = 'ap'
                    JOIN images i_lat ON i_lat.patient_id = p.id AND i_lat.view_type = 'lateral'
                    JOIN annotations a_ap ON a_ap.image_id = i_ap.id
                    JOIN annotations a_lat ON a_lat.image_id = i_lat.id
                )
            """).fetchone()[0]

            return {
                'total_patients': total_patients,
                'has_ap': stats[0],
                'has_lateral': stats[1],
                'ap_annotated': stats[2],
                'lateral_annotated': stats[3],
                'complete': complete,
            }
