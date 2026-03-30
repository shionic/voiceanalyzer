#!/usr/bin/env python3
"""
Metadata File Handler
Handles reading and writing metadata files for batch voice analysis processing.
Supports JSON format with per-file metadata overrides.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import csv


class MetadataEntry:
    """Represents a single file entry with metadata"""
    
    def __init__(self,
                 filepath: str,
                 author: Optional[str] = None,
                 author_source: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 reliable_quality_rating: Optional[float] = None,
                 unreliable_quality_rating: Optional[float] = None,
                 **extra_fields):
        """
        Initialize metadata entry.
        
        Args:
            filepath: Path to the audio file
            author: Speaker name/identifier
            author_source: Source of the recording
            tags: List of tags
            reliable_quality_rating: Reliable quality rating (0-1)
            unreliable_quality_rating: Unreliable quality rating (0-1)
            **extra_fields: Any additional fields to store
        """
        self.filepath = filepath
        self.author = author
        self.author_source = author_source
        self.tags = tags or []
        self.reliable_quality_rating = reliable_quality_rating
        self.unreliable_quality_rating = unreliable_quality_rating
        self.extra_fields = extra_fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            'filepath': self.filepath,
            'author': self.author,
            'author_source': self.author_source,
            'tags': self.tags,
            'reliable_quality_rating': self.reliable_quality_rating,
            'unreliable_quality_rating': self.unreliable_quality_rating,
        }
        data.update(self.extra_fields)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataEntry':
        """Create from dictionary"""
        filepath = data.pop('filepath')
        return cls(filepath=filepath, **data)
    
    def __repr__(self):
        return f"MetadataEntry(filepath='{self.filepath}', author='{self.author}')"


class MetadataFile:
    """Handler for metadata files"""
    
    def __init__(self, filepath: str):
        """
        Initialize metadata file handler.
        
        Args:
            filepath: Path to the metadata file
        """
        self.filepath = Path(filepath)
        self.entries: List[MetadataEntry] = []
    
    def read(self) -> List[MetadataEntry]:
        """
        Read metadata file and return list of entries.
        
        Returns:
            List of MetadataEntry objects
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.filepath}")
        
        ext = self.filepath.suffix.lower()
        
        if ext == '.json':
            self.entries = self._read_json()
        elif ext == '.jsonl':
            self.entries = self._read_jsonl()
        elif ext == '.csv':
            self.entries = self._read_csv()
        else:
            raise ValueError(f"Unsupported metadata file format: {ext}. "
                           f"Supported formats: .json, .jsonl, .csv")
        
        return self.entries
    
    def _read_json(self) -> List[MetadataEntry]:
        """Read JSON metadata file"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both array of objects and single object with files array
        if isinstance(data, list):
            entries_data = data
        elif isinstance(data, dict) and 'files' in data:
            entries_data = data['files']
        else:
            raise ValueError("JSON must be an array or object with 'files' array")
        
        return [MetadataEntry.from_dict(entry) for entry in entries_data]
    
    def _read_jsonl(self) -> List[MetadataEntry]:
        """Read JSONL (JSON Lines) metadata file"""
        entries = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(MetadataEntry.from_dict(data))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
        
        return entries
    
    def _read_csv(self) -> List[MetadataEntry]:
        """Read CSV metadata file"""
        entries = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                if 'filepath' not in row:
                    print(f"Warning: Skipping row {row_num} - missing 'filepath' column")
                    continue
                
                # Convert string tags to list
                if 'tags' in row and row['tags']:
                    row['tags'] = [t.strip() for t in row['tags'].split(',')]
                
                # Convert numeric fields
                for field in ['reliable_quality_rating', 'unreliable_quality_rating']:
                    if field in row and row[field]:
                        try:
                            row[field] = float(row[field])
                        except ValueError:
                            row[field] = None
                
                # Remove empty values
                row = {k: v for k, v in row.items() if v not in ('', None)}
                
                entries.append(MetadataEntry.from_dict(row))
        
        return entries
    
    def write(self, entries: List[MetadataEntry], format: str = 'json'):
        """
        Write metadata to file.
        
        Args:
            entries: List of MetadataEntry objects
            format: Output format ('json', 'jsonl', 'csv')
        """
        self.entries = entries
        
        if format == 'json':
            self._write_json()
        elif format == 'jsonl':
            self._write_jsonl()
        elif format == 'csv':
            self._write_csv()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _write_json(self):
        """Write JSON metadata file"""
        data = [entry.to_dict() for entry in self.entries]
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _write_jsonl(self):
        """Write JSONL metadata file"""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            for entry in self.entries:
                json.dump(entry.to_dict(), f, ensure_ascii=False)
                f.write('\n')
    
    def _write_csv(self):
        """Write CSV metadata file"""
        if not self.entries:
            return
        
        # Collect all possible fields
        all_fields = set()
        for entry in self.entries:
            all_fields.update(entry.to_dict().keys())
        
        fieldnames = ['filepath', 'author', 'author_source', 'tags',
                     'reliable_quality_rating', 'unreliable_quality_rating']
        # Add any extra fields
        extra_fields = sorted(all_fields - set(fieldnames))
        fieldnames.extend(extra_fields)
        
        with open(self.filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in self.entries:
                row = entry.to_dict()
                # Convert tags list to comma-separated string
                if 'tags' in row and isinstance(row['tags'], list):
                    row['tags'] = ', '.join(row['tags'])
                writer.writerow(row)
    
    def merge_with_defaults(self, 
                           default_author: Optional[str] = None,
                           default_source: Optional[str] = None,
                           default_tags: Optional[List[str]] = None,
                           default_reliable_quality: Optional[float] = None,
                           default_unreliable_quality: Optional[float] = None) -> List[MetadataEntry]:
        """
        Merge metadata entries with default values.
        File-specific values take precedence over defaults.
        
        Args:
            default_author: Default author
            default_source: Default source
            default_tags: Default tags
            default_reliable_quality: Default reliable quality rating
            default_unreliable_quality: Default unreliable quality rating
            
        Returns:
            List of merged MetadataEntry objects
        """
        merged_entries = []
        
        for entry in self.entries:
            # Create merged entry - file values take precedence
            merged = MetadataEntry(
                filepath=entry.filepath,
                author=entry.author or default_author,
                author_source=entry.author_source or default_source,
                tags=entry.tags if entry.tags else (default_tags or []),
                reliable_quality_rating=(
                    entry.reliable_quality_rating 
                    if entry.reliable_quality_rating is not None 
                    else default_reliable_quality
                ),
                unreliable_quality_rating=(
                    entry.unreliable_quality_rating 
                    if entry.unreliable_quality_rating is not None 
                    else default_unreliable_quality
                ),
                **entry.extra_fields
            )
            merged_entries.append(merged)
        
        return merged_entries
    
    @staticmethod
    def create_template(filepath: str, format: str = 'json', sample_files: List[str] = None):
        """
        Create a template metadata file.
        
        Args:
            filepath: Output file path
            format: Format ('json', 'jsonl', 'csv')
            sample_files: Optional list of sample file paths
        """
        sample_files = sample_files or ["audio1.wav", "audio2.wav"]
        
        entries = [
            MetadataEntry(
                filepath=f,
                author="Speaker Name",
                author_source="interview",
                tags=["tag1", "tag2"],
                reliable_quality_rating=0.85,
                unreliable_quality_rating=0.70
            )
            for f in sample_files
        ]
        
        mf = MetadataFile(filepath)
        mf.write(entries, format=format)
        print(f"Template metadata file created: {filepath}")


def validate_metadata_entries(entries: List[MetadataEntry], base_dir: Optional[Path] = None) -> List[str]:
    """
    Validate metadata entries and return list of errors.
    
    Args:
        entries: List of MetadataEntry objects
        base_dir: Optional base directory used to resolve relative filepaths
        
    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    
    for i, entry in enumerate(entries, 1):
        # Check filepath exists
        if not entry.filepath:
            errors.append(f"Entry {i}: Missing filepath")
            continue
        
        filepath = Path(entry.filepath)
        if not filepath.is_absolute() and base_dir is not None:
            filepath = base_dir / filepath
        if not filepath.exists():
            errors.append(f"Entry {i}: File not found: {entry.filepath}")
        
        # Validate quality ratings
        for field, value in [
            ('reliable_quality_rating', entry.reliable_quality_rating),
            ('unreliable_quality_rating', entry.unreliable_quality_rating)
        ]:
            if value is not None and not (0.0 <= value <= 1.0):
                errors.append(
                    f"Entry {i}: {field} must be between 0 and 1, got {value}"
                )
    
    return errors


# Example usage
if __name__ == "__main__":
    # Create template
    MetadataFile.create_template("template.json", format='json')
    MetadataFile.create_template("template.csv", format='csv')
    
    print("\nTemplate files created!")
    print("Edit them to add your file metadata.")
