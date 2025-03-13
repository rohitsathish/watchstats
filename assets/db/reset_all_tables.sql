PRAGMA foreign_keys = OFF;
SELECT 'DELETE FROM ' || name || ';' FROM sqlite_master WHERE type = 'table';
PRAGMA foreign_keys = ON;