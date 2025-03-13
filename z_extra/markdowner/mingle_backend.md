# Project Files for mingle_backend

# Project Structure
```
└── mingle_backend
    ├── chrome_profile
    ├── config.py (280 lines)
    ├── db_handler
    │   ├── test.py (60 lines)
    │   ├── upload.py (298 lines)
    │   └── validate_events.py (71 lines)
    ├── llm_parse.py (42 lines)
    ├── llm_system_prompts
    │   └── project_prompt.md (157 lines)
    ├── main.py (47 lines)
    ├── messages
    │   └── messages_20250205_1843.json (75 lines)
    ├── messages_handler.py (105 lines)
    ├── parsed_events
    ├── README.md (35 lines)
    ├── requirements.txt (3 lines)
    ├── scraper.py (719 lines)
    ├── temp
    └── whatsapp_day_to_date.py (48 lines)
```
# File Contents
-----

FILE: config.py
CONTENT:
```py
# %%
import os
from typing import Dict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Directory for message files
MESSAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "messages")

# Time deltas
NEW_FILE_THRESHOLD = timedelta(hours=12)  # Create new file if last file is older than this
GROUP_SCRAPE_THRESHOLD = timedelta(hours=12)  # Rescrape group if last scrape is older than this


# Number of days to look back for messages
DAYS_BACK_TO_PROCESS = 7

# Group configurations with chatter flag
GROUPS: Dict[str, Dict[str, bool]] = {
    "HUMAN LIBRARY BENGALURU": {"chatter": False},
    "Copper + Cloves Happenings": {"chatter": False},
    "the STUDIO by Copper + Cloves": {"chatter": False},
    "The Reading Social": {"chatter": False},
    "Read A Kitaab: Bangalore": {"chatter": True},
    "BCC Community": {"chatter": False},
    "HSR Meetups Official": {"chatter": False},
    "BLR Events Hub": {"chatter": False},
    "Events BLR-BYOB": {"chatter": False},
    "No Pressure Improv!": {"chatter": False},
    "Courtyard Community": {"chatter": False},
    "New Acropolis Community": {"chatter": False},
    "Fit Club Bengaluru": {"chatter": True},
    "Sustainability101": {"chatter": True},
    "Terra.do Bangalore": {"chatter": True},
    "The Parallel Cinema Club": {"chatter": False},
    "Science Gallery Bengaluru": {"chatter": False},
    "Bookmarks Lahe Lahe": {"chatter": True},
    "Sakura Screenings": {"chatter": False},
    "BangaloreDrumsCollective": {"chatter": False},
    "Atta Galatta - Events": {"chatter": False},
    "Improv Lore Community": {"chatter": False},
    "Underline Center": {"chatter": False},
    "The Mango Tree": {"chatter": False},
    "The UnListed Club": {"chatter": False},
    "Bangalore IRLs": {"chatter": False},
    "Bengaluru Social": {"chatter": False},
    "MOSAMBI: Bengaluru": {"chatter": True},
    "MAP Youth Collective": {"chatter": False},
    "Dialogues Friends": {"chatter": False},
    "Putting Scene - 15": {"chatter": False},
    "Ekta's Gatherings": {"chatter": False},
    "SanimaWaale": {"chatter": True, "announce_gp": True},
}

# to extend - BYOB-Bangalore

# "Atta Galatta Book Club": {"chatter": True},

# Yet to add board gaming groups

PRIVATE_GROUPS: Dict[str, Dict[str, bool]] = {
    "Poker club HSR": {"chatter": True},
    "FCB - Events": {"chatter": True},
    "TGIF Ultimate Pick Up": {"chatter": True},
    "Bengaluru Picklers": {"chatter": True},
    "Bengaluru Foodies Community": {"chatter": False},
}

# Output file path
MESSAGES_JSON_PATH = os.path.join(
    MESSAGES_DIR, f"messages_{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y%m%d_%H%M')}.json"
)

# Event-related patterns for filtering messages in verbose groups
EVENT_PATTERNS = [
    # Core event indicators
    r"event",  # Generic event mention
    r"invite",
    r"meet-?up",  # Catches both meetup and meet-up
    r"last\s+chance",
    r"join\s+us",
    r"hosting",
    r"happening",
    r"save\s+the\s+date",
    r"mark\s+your\s+calendar",
    r"club",
    r"(?:this|next|coming)\s+weekend",
    # Time patterns (more specific)
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)",  # Matches time formats like "7pm", "7:30pm"
    r"this\s+(?:sat|sun|mon|tue|wed|thu|fri)(?:urday|day)?",  # Day mentions
    r"next\s+(?:sat|sun|mon|tue|wed|thu|fri)(?:urday|day)?",
    r"tomorrow\s+(?:evening|morning|afternoon)",
    # Registration/RSVP
    r"register\s+(?:here|now|at|via)",
    r"registration\s+(?:open|closes?|link)",
    r"rsvp",
    r"limited\s+(?:seats|spots)",
    # Event types (expanded)
    r"workshop",
    r"meetup",
    r"session",
    r"screening",
    r"performance\s+(?:by|of|at)",
    r"concert",
    r"exhibition",
    r"talk\s+(?:by|on|about)",
    r"seminar",
    r"discussion",
    r"panel\s+(?:on|about)",
    r"showcase\s+(?:of|by)",
    r"open\s+mic",
    r"jam\s+session",
    r"book\s+reading",
    r"art\s+(?:show|exhibition)",
    r"dance\s+(?:class|workshop|performance)",
    r"music\s+(?:show|night|performance)",
    r"poetry\s+(?:reading|session)",
    r"film\s+(?:screening|showing)",
    r"game\s+(?:night|session)",
    r"comedy\s+(?:show|night)",
    r"quiz\s+(?:night|competition)",
    r"food\s+(?:tasting|workshop)",
    r"networking\s+(?:event|session)",
    r"masterclass\s+(?:on|about|with)",
    r"live\s+(?:music|performance|show)",
    r"improv\s+(?:show|night|session)",
    # Entry/Tickets
    r"entry\s+(?:fee|is|:)",
    r"tickets?\s+(?:at|available|:)",
]

# DOM Selectors for WhatsApp Web Elements
SELECTORS = {
    "ARCHIVED_TEXT": 'div:text-is("Archived")',
    "ARCHIVED_HEADER": 'h1:has-text("Archived")',
    "GROUP_CONTAINER": 'div[role="group"]',
    "ARCHIVED_GROUPS": '//h1[text()="Archived"]/ancestor::header/following-sibling::div//div[@role="gridcell"]//span[@dir="auto"]',
    "CHAT_CONTAINER": "main",
    "SCROLL_CONTAINER": '//h1[text()="Archived"]/ancestor::header/following-sibling::div',
    "CHAT_SCROLL_BOTTOM_BUTTON": 'div[role="button"][aria-label="Scroll to bottom"]',
    "READ_MORE_BUTTON": 'div[role="button"]:text-is("Read more")',
    "MESSAGE_DATE_DIVS": 'div[role="application"] > div[tabindex="-1"] > div > span[dir="auto"]',
    "OLDER_MESSAGES_BUTTON": ':text-matches("click here to get older messages|Use WhatsApp on your phone to see older")',
    "CHAT_SCROLL_CONTAINER": 'div[id="main"] > div > div[class*="copyable-area"] > div[tabindex="0"]',
    "MESSAGE_CONTAINER": 'div[role="application"] > div',
    "MESSAGE_TEXT": 'span[dir="ltr"]',
    "MESSAGE_EVENT": 'div[aria-label*="Event"]',
    "MESSAGE_DATE_HEADER": 'span[dir="auto"]',
}

CATEGORY_TAG_LIST = {
    "Reading & Literature": [
        "reading",
        "books",
    ],
    "Speaking & Discussion": [
        "debate",
        "storytelling",
        "discussion"
    ],
    "Movies & Screenings": [
        "movie night",
        "screening",
        "short films"
    ],
    "Performing Arts": [
        "theatre",
        "drama",
        "dance",
    ],
    "Music & Concerts": [
        "live music",
        "music festival",
    ],
    "Socializing & Meetups": [
        "social mixer",
        "networking event",
        "singles mixer",
        "meet & greet",
    ],
    "Food & Beverage": [
        "dining",
        "brunch",
        "lunch",
        "dinner",
    ],
    "Board Games & Tabletop": [
        "board games",
        "d&d",
        "dungeons & dragons",
        "chess meetup"
    ],
    "Open Mic & Poetry": [
        "poetry jam",
        "open mic night"
    ],
    "Learning & Education": [
        "workshop",
        "class",
        "course",
        "training",
    ],
    "Sustainability & Environment": [
        "sustainability",
        "climate",
    ],
    "Community Service & Volunteering": [
        "volunteering",
        "charity",
        "fundraiser",
        "ngo event",
        "awareness drive"
    ],
    "Sports & Fitness": [
        "sports",
        "running",
        "cycling",
        "fitness",
        "workout",
        "yoga",
        "zumba",
        "marathon",
        "climbing",
        "martial arts",
    ],
    "Culture & Heritage": [
        "cultural event",
        "heritage walk",
    ],
    "Business & Entrepreunership": [
        "networking",
        "business meetup",
        "startup event",
        "conference",
        "hackathon",
    ],
    "Kids & Family": [
        "kids event",
        "children’s workshop",
        "family event",
        "family-friendly",
        "kids activity",
        "parenting workshop"
    ],
    "Art & Craft": [
        "art fair",
        "craft fair",
        "exhibition",
        "art show",
        "handicrafts",
        "artisan market",
        "art gallery"
    ],
    "Health & Wellness": [
        "mental health",
        "yoga",
        "meditation"
    ],
    "Travel & Adventure": [
        "hike",
        "trip",
    ],
    "Science & Tech": [
        "science lecture",
        "science gallery"
    ],
    "Religion & Spirituality": [
        "astrology",
        "tarot reading",
        "bhajan"
    ],
    "Pets & Animals": [
        "pet adoption",
        "wildlife retreat"
    ]
}

#%%

list(CATEGORY_TAG_LIST.keys())
```
-----

FILE: db_handler\test.py
CONTENT:
```py
# %%
import json
from pathlib import Path
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple


# %%
def check_event_id_duplicates(date_str: str) -> Tuple[bool, Dict[str, List[dict]]]:
    """
    Check for duplicate event_ids in a parsed events file.

    Args:
        date_str: Date string in format YYYYMMDD

    Returns:
        Tuple of (has_duplicates, duplicate_dict)
        where duplicate_dict maps event_ids to list of events with that id
    """
    script_path = Path(__file__).resolve()
    parsed_dir = script_path.parent.parent / "parsed_events"
    events_file = parsed_dir / f"events_{date_str}.json"

    with events_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        events = data.get("events", [])

    # Track events by their generated id
    events_by_id = defaultdict(list)

    for event in events:
        # Generate event_id using same logic as uploader
        source = "WhatsApp"
        source_group = event.get("source_whatsapp_group", "")
        post_datetime = event.get("post_datetime", "")
        id_string = f"{source}_{source_group}_{post_datetime}"
        event_id = hashlib.md5(id_string.encode()).hexdigest()[:16]

        events_by_id[event_id].append(event)

    # Filter to only duplicates
    duplicates = {k: v for k, v in events_by_id.items() if len(v) > 1}

    return bool(duplicates), duplicates


# %%
# Example usage
date_str = "20250127"
has_dupes, dupes = check_event_id_duplicates(date_str)
print(f"Has duplicates: {has_dupes}")
if has_dupes:
    for event_id, events in dupes.items():
        print(f"\nDuplicate ID: {event_id}")
        for i, event in enumerate(events, 1):
            print(f"Event {i}:")
            print(f"  Name: {event.get('name')}")
            print(f"  Source Group: {event.get('source_whatsapp_group')}")
            print(f"  Post DateTime: {event.get('post_datetime')}")
```
-----

FILE: db_handler\upload.py
CONTENT:
```py
#!/usr/bin/env python3
"""
Event uploader for WhatsApp events to Supabase.

This script reads JSON events from parsed_events, transforms them
to match the PostgreSQL schema, and does a single batch upsert via PostgREST.
"""

import os
import sys
import json
import hashlib
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env


class EventUploader:
    """Handles event transformation and bulk upload to Supabase via PostgREST."""

    def __init__(self) -> None:
        self.db_url = os.getenv("DB_URL")  # e.g. https://<project>.supabase.co
        self.anon_key = os.getenv("ANON_KEY")  # supabase anon or service role key
        self.gmaps_apikey = os.getenv("GMAPS_APIKEY")

        # Quick check for required env variables
        if not self.db_url or not self.anon_key or not self.gmaps_apikey:
            raise ValueError("Missing DB_URL, ANON_KEY, or GMAPS_APIKEY environment variables.")

        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @staticmethod
    def read_json_file(file_path: str) -> Dict[str, Any]:
        """Reads JSON from the specified file path and returns it as a dict."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def validate_required_fields(self, event: Dict[str, Any]) -> Optional[str]:
        """Check presence and basic type of mandatory fields for minimal viability."""
        required = {"name": str, "source_whatsapp_group": str, "post_datetime": str}
        for field, ftype in required.items():
            if field not in event:
                return f"Missing required field: {field}"
            if not isinstance(event[field], ftype):
                return f"Invalid type for {field}: expected {ftype.__name__}"
        return None

    def validate_data_types(self, e: Dict[str, Any]) -> Optional[str]:
        """Validate each field to ensure it matches the DB's schema expectations."""
        # Arrays
        array_fields = ["categories", "tags"]
        for field in array_fields:
            val = e.get(field)
            if val is not None:
                if not isinstance(val, list):
                    return f"'{field}' must be a list"
                if any(not isinstance(item, str) for item in val):
                    return f"All items in '{field}' must be strings"

        # Numeric: cost
        if "cost" in e and e["cost"] is not None:
            try:
                float(e["cost"])  # just test convert
            except (ValueError, TypeError):
                return "'cost' must be numeric"

        # Booleans
        bool_fields = ["is_update", "is_cancelled"]
        for bf in bool_fields:
            if bf in e and e[bf] not in [True, False, None]:
                return f"'{bf}' must be boolean"

        # Datetime
        datetime_fields = ["post_datetime", "start_datetime", "end_datetime"]
        for dt in datetime_fields:
            val = e.get(dt)
            if val:
                try:
                    datetime.fromisoformat(val.replace("Z", "+00:00"))
                except ValueError:
                    return f"'{dt}' must be a valid ISO datetime"

        # links must be JSON-serializable
        if "links" in e and e["links"] is not None:
            try:
                json.dumps(e["links"])
            except (TypeError, ValueError):
                return "'links' must be JSON-serializable"

        return None

    async def fetch_place_details(self, address: str) -> Optional[Dict[str, Any]]:
        """Use Google Place TextSearch for the address to get place_id, lat/lng, and formatted_address."""
        if not address or not self.session:
            return None

        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query": address,
            "key": self.gmaps_apikey,
            "fields": "place_id,geometry/location,formatted_address"  # Specify required fields here
        }
        try:
            async with self.session.get(url, params=params, timeout=10) as resp:
                data = await resp.json()
                if data.get("status") != "OK" or not data.get("results"):
                    print(f"TextSearch returned {data.get('status')} for '{address}'")
                    return None

                top_result = data["results"][0]
                place_id = top_result["place_id"]
                lat = top_result["geometry"]["location"]["lat"]
                lng = top_result["geometry"]["location"]["lng"]
                formatted_addr = top_result.get("formatted_address")

                return {
                    "place_id": place_id,
                    "lat": lat,
                    "lng": lng,
                    "formatted_address": formatted_addr
                }
        except Exception as ex:
            print(f"Error in fetch_place_details for '{address}': {ex}")
            return None


    @staticmethod
    def generate_event_id(evt: Dict[str, Any]) -> str:
        """Generate stable event_id by hashing e.g. post_datetime + source_whatsapp_group."""
        raw = f"{evt['name']}_WhatsApp_{evt['source_whatsapp_group']}_{evt['post_datetime']}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    async def transform_event(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a raw event into a fully qualified DB row dict."""
        # 1. Check required fields
        missing_err = self.validate_required_fields(raw)
        if missing_err:
            raise ValueError(missing_err)

        # 2. Type check
        type_err = self.validate_data_types(raw)
        if type_err:
            raise ValueError(type_err)

        # 3. Generate event_id
        ev_id = self.generate_event_id(raw)

        # 4. Default booleans
        is_update = raw.get("is_update", False)
        is_cancelled = raw.get("is_cancelled", False)

        # 5. cost
        cost_val = raw.get("cost")
        if cost_val is not None:
            try:
                cost_val = float(cost_val)
            except (ValueError, TypeError):
                cost_val = None

        # 6. Attempt geocoding
        llm_address = raw.get("venue_address_llm")
        place_info = None
        if llm_address:
            place_info = await self.fetch_place_details(llm_address)

        lat_long = None
        gmaps_link = None
        gmaps_address = None

        if place_info:
            lat_long = f"SRID=4326;POINT({place_info['lng']} {place_info['lat']})"
            gmaps_link = f"https://www.google.com/maps/place/?q=place_id:{place_info['place_id']}"
            gmaps_address = place_info.get("formatted_address")

        return {
            "event_id": ev_id,
            "name": raw["name"],
            "categories": raw.get("categories", []),
            "tags": raw.get("tags", []),
            "summary": raw.get("summary"),
            "notes_to_admin": raw.get("notes_to_admin"),
            "cost": cost_val,
            "source_whatsapp_group": raw["source_whatsapp_group"],
            "source": "WhatsApp",
            "area": raw.get("area"),
            "venue_address_llm": llm_address,
            "venue_address_gmaps": gmaps_address,
            "post_datetime": raw.get("post_datetime"),
            "start_datetime": raw.get("start_datetime"),
            "end_datetime": raw.get("end_datetime"),
            "is_update": is_update,
            "is_cancelled": is_cancelled,
            "links": raw.get("links", {}),
            "lat_long": lat_long,
            "gmaps_link": gmaps_link,
        }

    async def upload_events(self, events: List[Dict[str, Any]]) -> None:
        """Transforms each event, then batch POST to the 'events' table with upsert-like logic."""
        if not self.session:
            raise RuntimeError("No HTTP session available. Must be in async context.")

        # Transform all events
        final_list: List[Dict[str, Any]] = []
        for item in events:
            row = await self.transform_event(item)
            final_list.append(row)

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.anon_key}",
            "apikey": self.anon_key,
            "Content-Type": "application/json",
            # "on_conflict=event_id" might be needed in your Supabase config or
            # you can rely on primary key conflict. Usually "resolution=merge-duplicates" or "upsert".
            "Prefer": "resolution=merge-duplicates",
        }

        url = f"{self.db_url}/rest/v1/events"

        try:
            async with self.session.post(url, headers=headers, json=final_list, timeout=30) as resp:
                if resp.status not in (200, 201):
                    error_text = await resp.text()
                    raise ValueError(f"Batch upload failed (status {resp.status}): {error_text}")
                print(f"Successfully uploaded {len(final_list)} events.")
        except Exception as ex:
            raise RuntimeError(f"Error during batch upload: {ex}")

    def validate_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Checks only the presence of minimal required fields for each event."""
        errors = []
        for idx, e in enumerate(events):
            res = self.validate_required_fields(e)
            if res:
                errors.append({"index": idx, "name": e.get("name"), "error": res})
        return errors


async def main_async():
    """Async main entrypoint."""
    if len(sys.argv) < 2:
        print("Usage: python upload.py YYYYMMDD [YYYYMMDD ...]")
        sys.exit(1)

    date_args = sys.argv[1:]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parsed_dir = os.path.join(os.path.dirname(script_dir), "parsed_events")

    async with EventUploader() as uploader:
        for date_str in date_args:
            path = os.path.join(parsed_dir, f"events_{date_str}.json")
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            data = uploader.read_json_file(path)
            if "events" not in data:
                print(f"No 'events' key in {path}")
                continue

            ev_list = data["events"]
            errs = uploader.validate_events(ev_list)
            if errs:
                print(f"\nValidation errors in {path}:")
                for err in errs:
                    print(f"  Event #{err['index']} ({err['name']}): {err['error']}")
                print("Aborting upload due to errors.")
                continue

            print(f"\nProcessing {len(ev_list)} events from {path} ...")
            try:
                await uploader.upload_events(ev_list)
            except Exception as exc:
                print(f"Error uploading events from {path}: {exc}")
                sys.exit(1)


def main():
    """Sync wrapper for the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
```
-----

FILE: db_handler\validate_events.py
CONTENT:
```py
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

def validate_event(event: Dict) -> Optional[str]:
	"""Validate a single event and return error message if invalid"""
	required_fields = {
		'name': str,
		'source_whatsapp_group': str,
		'post_datetime': str
	}

	for field, field_type in required_fields.items():
		if field not in event:
			return f"Missing required field: {field}"
		if not isinstance(event[field], field_type):
			return f"Invalid type for {field}: expected {field_type}, got {type(event[field])}"

	return None

def generate_event_id(event: Dict) -> str:
	"""Generate a shorter event ID using MD5 instead of SHA-256"""
	source = "WhatsApp"
	id_string = f"{source}_{event['source_whatsapp_group']}_{event['post_datetime']}"
	return hashlib.md5(id_string.encode()).hexdigest()[:16]  # Using first 16 chars of MD5

def validate_events_file(file_path: Path) -> List[Dict]:
	"""Validate all events in a file and return list of validation errors"""
	with open(file_path) as f:
		data = json.load(f)

	validation_errors = []

	for idx, event in enumerate(data['events']):
		error = validate_event(event)
		if error:
			validation_errors.append({
				'event_index': idx,
				'event_name': event.get('name', 'Unknown'),
				'error': error
			})

	return validation_errors

def main():
	events_dir = Path(__file__).parent.parent / 'parsed_events'
	all_errors = []

	for file_path in events_dir.glob('events_*.json'):
		errors = validate_events_file(file_path)
		if errors:
			all_errors.append({
				'file': file_path.name,
				'errors': errors
			})

	if all_errors:
		print("Validation errors found:")
		for file_errors in all_errors:
			print(f"\nFile: {file_errors['file']}")
			for error in file_errors['errors']:
				print(f"  Event #{error['event_index']}: {error['event_name']}")
				print(f"    Error: {error['error']}")
		return False

	print("All events validated successfully!")
	return True

if __name__ == "__main__":
	main()
```
-----

FILE: llm_parse.py
CONTENT:
```py
# %% Imports and Setup
import os
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph
from pydantic import BaseModel
from typing import List, Optional

# Load environment variables
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_APIKEY")


# %% Define Schema
class Message(BaseModel):
    sender: str
    content: str
    timestamp: Optional[str] = None
    mentions: Optional[List[str]] = None


class ChatData(BaseModel):
    messages: List[Message]
    chat_name: Optional[str] = None


# %% Configure Graph
graph_config = {"llm": {"api_key": GEMINI_KEY, "model": "gemini-pro"}}

scraper = SmartScraperGraph(
    prompt="Extract message information including sender, content, and any mentions. Format timestamps if present.",
    config=graph_config,
    schema=ChatData,
)

# %% Test with sample data
sample_text = """
[2:45 PM] John: Hey @Alice, can you review the document?
[2:46 PM] Alice: Sure @John, I'll take a look now
"""

result = scraper.process(sample_text)
print(result)
```
-----

FILE: llm_system_prompts\project_prompt.md
CONTENT:
```md
# Mingle Scraper - Low Level Design (LLD)

## Part 1: Overall Project Approach

### 1.1 Project Context

- **Objective**: Scrape WhatsApp group messages to capture event-relevant data.
- **Output**: A `messages.json` file containing messages from specific groups. Each message includes text, timestamp, and derived date.
- **days_back_to_process**: The script only loads messages newer than `current_date - days_back_to_process`.
- **Existing Code**: Reference `temp/app_old.py` to start with and get relevant DOM elements and pointers. Leave a comment for when the user needs to input the right DOM element for it to work - start comment with "USER INPUT HERE".
- **Testing Restrictions**: Do not test solutions through terminal or sidecar, since you do not have the visual feedback of the browser.

### 1.2 Key Enhancements

1. **Message Datetime Storage**
   - For every message, store an **actual datetime** (e.g., `YYYY-MM-DD HH:MM`).
   - Stopping Condition: If the earliest loaded message is older than `days_back_to_process`, stop scrolling.

2. **Regex‑Based Filtering (Selective)**
   - For groups flagged as `chatter=True`, apply **basic regex** to remove irrelevant chatter.
   - For other groups, store **all** messages without filtering.

### 1.3 Overall Flow

1. **Browser Automation**
   - Use a headless or visible browser (Playwright recommended).
   - Log into WhatsApp Web, open the group list, scroll as needed.
2. **Group-by-Group Scraping**
   - For each group in `groups_list`:
     - If last scrape was recent, skip. Otherwise, parse messages from the newest to older.
     - Expand “Read more” messages, handle any spinner/loader.
3. **Data Output**
   - Append or merge the results into `messages.json`.
   - If the group is `chatter=True`, run regex on each message to exclude known “noise.”
4. **Stop Condition**
   - Once messages older than `days_back_to_process` are encountered, stop upward scroll.
5. **Next Steps**
   - The coded solution will integrate with future classification or LLM parsing if desired.

---

## Part 2: Stepwise Implementation & User Testing

Below is a **module-by-module** (step-by-step) plan. The LLM coding agent should **implement** each step and then **prompt a user test** before proceeding.

### Step 1: Environment Setup
**Instructions**
1. Create a Python project with `requirements.txt`.
2. Install automation dependencies (e.g. Playwright).
3. Create initial files:
   - `main.py` (entry point)
   - `scraper.py` (primary logic)
   - `config.py` (for future environment variables)
   - `whatsapp_day_to_date.py` (function for date conversion)
4. Add a minimal `README.md` describing how to install and run.

**User Test**
- Confirm the environment can be installed via `pip install -r requirements.txt`.
- Confirm `main.py` runs a simple “Hello from Mingle Scraper!” message.

---

### Step 2: WhatsApp Web Login Flow
**Instructions**
1. Implement a `login_whatsapp()` function in `scraper.py` that:
   - Launches the browser (headless or interactive).
   - Navigates to WhatsApp Web.
   - Waits for user to scan QR if needed (e.g., detecting the QR DOM element).
   - Persists session (cookies/user-data-dir) to skip re-login next time.
2. In `main.py`, call `login_whatsapp()` and log success or error messages.

**User Test**
- Run `main.py`, watch the browser open WhatsApp Web.
- Scan QR code if prompted.
- Confirm console logs “Login complete” or “Session loaded.”

---

### Step 3: Group Discovery & Scrolling
**Instructions**
1. **Left Panel Scroll**: In `scraper.py`, build a function `find_groups(groups_list)` that:
   - Scrolls the left chat list.
   - For each visible group name, compare with items in `groups_list` (string contains).
   - For each match, store the DOM element reference or clickable selector.
2. Ensure multiple passes: keep scrolling if more groups might be off-screen until all are discovered or we reach the end.
3. If a group was recently scraped (based on `messages.json` metadata), skip.

**User Test**
- Add 1–2 real group names to `groups_list`.
- Run and verify logs: “Found group: <GroupName>” or “Skipped group due to recent scrape.”

---

### Step 4: Open Group & Scroll to Bottom
**Instructions**
1. In a new function, `open_group_and_preload(group_name)`:
   - Click the group in the left panel.
   - Wait for loading spinner (if any) to disappear (max 5–10 min).
   - Force scroll to bottom repeatedly until the position is stable (detect no further scrolling).
2. Return control once the bottom is confirmed loaded.

**User Test**
- User picks a group with unread messages.
- Script reports: “Spinner hidden, scrolled to bottom successfully.”

---

### Step 5: Extract & Scroll Up Messages
**Instructions**
1. Write `scroll_and_collect_messages(group_name)` that does:
   - Start from bottom, gather visible messages.
   - For each message, record `sender`, `text`, **datetime** from the DOM (converting day headers with `whatsapp_day_to_date`).
   - If a “Read more” button is detected, click and wait.
   - If “Syncing older messages” appears, wait up to 5 minutes.
   - Identify the topmost message date in the viewport. If it’s **older** than `(today - days_back_to_process)`, break out.
   - Otherwise, scroll up further (by y-axis or chunk approach) and repeat.
2. Return the newly collected messages.

**User Test**
- User tries with a group known to have older messages.
- Confirm the script stops once it hits messages older than `(today - days_back_to_process)`.
- Check if partial messages are in memory or logged.

---

### Step 6: Regex Filtering & Data Persistence
**Instructions**
1. If `chatter=true` for this group, apply **basic regex** on each message to exclude noise.
2. Append the final, filtered message list to `messages.json`:
   - If file doesn’t exist, create it.
   - If it exists, read it, merge deduplicating as needed.
3. Add metadata: e.g. `"scrape_datetime": <current UTC time>`.

**User Test**
- Mark one group as `chatter=true` in `groups_list`.
- Confirm the script logs “Applying regex filter.”
- Inspect `messages.json`: confirm that obviously non-event lines are missing, while others remain.
- Confirm all messages have a `datetime` field.

---

### Step 7: Final Testing & Cleanup
**Instructions**
1. Do a final check on the entire pipeline.
   - `login_whatsapp()`, `find_groups(...)`, `open_group_and_preload(...)`, `scroll_and_collect_messages(...)`, `regex_filter(...)`, `append_to_json(...)`.
2. Add docstrings to all major functions.
3. Print summary stats (e.g., total messages scraped, groups processed).

**User Test**
- Run a full pass on multiple groups.
- Verify final `messages.json` includes correct data, no errors encountered, and minimal chatter for `chatter=true` groups.

---

Rough Notes

1.
```
-----

FILE: main.py
CONTENT:
```py
import argparse
from scraper import WhatsAppScraper
from config import GROUPS


def main():
    """Main entry point for the WhatsApp scraper."""
    keep_open = True
    test_run = True
    days_back = 10

    print("Starting Mingle Scraper...")
    print(f"Test run: {'enabled' if test_run else 'disabled'}")
    print(f"Looking back {days_back} days for messages")

    # Create scraper instance with command line arguments
    scraper = WhatsAppScraper(keep_open=keep_open, test_run=test_run)
    scraper.start_playwright()

    try:
        if scraper.initialize():
            if scraper.is_port_in_use():
                print("Using existing Chrome session...")
            else:
                print("Starting new Chrome session...")

            if scraper.wait_for_login():
                print("WhatsApp Web login successful!")
                # Process groups with days_back parameter
                found_groups = scraper.find_groups(list(GROUPS.keys()), days_back)
                if found_groups:
                    print(f"Successfully processed groups: {found_groups}")
                else:
                    print("No groups were processed")
            else:
                print("Login failed")
        else:
            print("Failed to initialize WhatsApp Web")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        scraper.stop_playwright()
        print("Script completed")


if __name__ == "__main__":
    main()
```
-----

FILE: messages\messages_20250205_1843.json
-----

FILE: messages_handler.py
CONTENT:
```py
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
from zoneinfo import ZoneInfo
import glob
from unidecode import unidecode

from config import NEW_FILE_THRESHOLD, GROUP_SCRAPE_THRESHOLD, MESSAGES_DIR, MESSAGES_JSON_PATH


class MessagesHandler:
    """Handler for WhatsApp messages JSON operations."""

    def __init__(self):
        self.base_dir = MESSAGES_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_latest_messages_file(self) -> Optional[str]:
        pattern = os.path.join(self.base_dir, "messages_*.json")
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files, key=os.path.getctime)

    def _clean_message_text(self, text: str) -> str:
        """Clean message text while preserving single newlines."""
        if not text:
            return ""

        # Convert to ASCII-compatible form
        text = unidecode(text)

        # Remove certain invisible or special chars
        text = text.replace("\u200b", "")
        text = text.replace("\xa0", " ")
        text = text.replace("\t", " ")

        # Normalize form
        text = unicodedata.normalize("NFKC", text)

        # Standardize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Convert multiple newlines -> single newline
        text = re.sub(r"\n+", "\n", text)

        # Optionally strip leading/trailing newlines
        text = text.strip("\n")

        # Remove undesired control chars except newline
        cleaned_chars = []
        for ch in text:
            if unicodedata.category(ch)[0] != "C" or ch == "\n":
                cleaned_chars.append(ch)

        return "".join(cleaned_chars)

    def _clean_dict(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._clean_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_dict(item) for item in obj]
        elif isinstance(obj, str):
            return self._clean_message_text(obj)
        return obj

    def create_messages_file(self, groups_data: List[Dict]) -> None:
        """Create messages JSON file with cleaned data."""
        try:
            output_data = {
                "created_at": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%dT%H:%M%z"),
                "whatsapp_groups": groups_data,
            }
            cleaned_data = self._clean_dict(output_data)

            if not os.path.exists(MESSAGES_DIR):
                os.makedirs(MESSAGES_DIR, exist_ok=True)

            # We do NOT override iterencode to replace '\\n'.
            # Standard JSON will store them as \n (escaped newlines).
            with open(MESSAGES_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error creating messages file: {e}")
            raise

    def get_last_scrape_datetime(self) -> Optional[datetime]:
        """Get the created_at datetime from the latest messages file."""
        latest_file = self._get_latest_messages_file()
        if not latest_file:
            return None

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                created_at = data.get("created_at")
                if created_at:
                    return datetime.fromisoformat(created_at)
        except Exception as e:
            print(f"Error reading messages file: {e}")
        return None
```
-----

FILE: README.md
CONTENT:
```md
# Mingle Scraper

A WhatsApp Web scraper to capture event-relevant data from specified groups.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Playwright browsers:
```bash
playwright install chromium
```

## Usage

Run the scraper:
```bash
python main.py
```

The script will:
1. Open WhatsApp Web in a browser window
2. Prompt for QR code scan if needed
3. Scrape messages from configured groups
4. Save results to `messages.json`

## Configuration

Edit `config.py` to:
- Modify group list and chatter flags
- Adjust days to look back for messages
- Change output file path
```
-----

FILE: requirements.txt
CONTENT:
```txt
playwright==1.40.0
beautifulsoup4==4.12.2
python-dotenv==1.0.0
```
-----

FILE: scraper.py
CONTENT:
```py
# %%
from playwright.sync_api import sync_playwright, Page, Browser
from bs4 import BeautifulSoup
from typing import Optional, Dict, List
from messages_handler import MessagesHandler
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import time
import socket
import os
import re
import random


from config import GROUPS, MESSAGES_JSON_PATH, SELECTORS, EVENT_PATTERNS
from whatsapp_day_to_date import convert_whatsapp_date


class WhatsAppScraper:
    """WhatsApp Web scraper using Playwright."""

    def _verify_announcement_group(self, group_name: str) -> bool:
        """Verify if a group has the required Announcements span under any header tag.

        Args:
            group_name: Name of the group to verify

        Returns:
            bool: True if verification passes or not required, False otherwise
        """
        # Check if this group requires announcement verification
        group_config = GROUPS.get(group_name, {})
        if not group_config.get("announce_gp", False):
            return True  # No verification needed

        try:
            # Look for any header that has a descendant span with title="Announcements"
            announcement_span = self.page.locator('header span[title="Announcements"]').first
            return announcement_span.is_visible()
        except Exception as e:
            print(f"Error verifying announcement group: {e}")
            return False

    def __init__(self, debugging_port: int = 9222, keep_open: bool = False, test_run: bool = False):
        """Initialize the WhatsApp scraper."""
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.messages = {}
        self.debugging_port = debugging_port
        self.keep_open = keep_open
        self.test_run = test_run
        # Initialize messages handler
        self.messages_handler = MessagesHandler()

    def is_port_in_use(self) -> bool:
        """Check if Chrome debugging port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", self.debugging_port)) == 0

    def start_playwright(self):
        """Start the Playwright instance."""
        self.playwright = sync_playwright().start()

    def stop_playwright(self):
        """Stop or disconnect from the browser based on keep_open."""
        if self.keep_open:
            print("Browser will remain open. Press Enter to close...")
            input()

        # Fully close the browser & Playwright
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        print("Browser fully closed and Playwright stopped")

    def initialize(self) -> bool:
        """Initialize persistent browser session using a dedicated profile.
        Retains an existing session if the debugging port is in use.
        """
        try:
            if self.is_port_in_use():
                print(f"Connecting to existing Chrome session on port {self.debugging_port}")
                self.browser = self.playwright.chromium.connect_over_cdp(f"http://localhost:{self.debugging_port}")
                self.context = self.browser.contexts[0]
                self.page = self.context.pages[0]
            else:
                # Define a persistent profile folder to store all browser data
                profile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chrome_profile")
                print("Starting persistent Chrome session using profile:", profile_dir)

                # Launch a persistent context with flags to reduce stored data
                self.browser = self.playwright.chromium.launch_persistent_context(
                    profile_dir,
                    headless=False,
                    viewport=None,
                    args=[
                        f"--remote-debugging-port={self.debugging_port}",
                        "--disk-cache-size=0",  # Disable disk caching
                        "--disable-application-cache",  # Disable the application cache
                        "--disable-sync",  # Turn off profile syncing
                        "--disable-extensions",  # Prevent extensions from being loaded
                        "--no-sandbox",
                        "--start-maximized",  # Start maximized
                        "--disable-gpu",  # Disable GPU hardware acceleration
                    ],
                )

                # Use the first page if available, otherwise create a new one.
                if self.browser.pages:
                    self.page = self.browser.pages[0]
                else:
                    self.page = self.browser.new_page()

                try:
                    cdp = self.page.context.new_cdp_session(self.page)
                    window_info = cdp.send("Browser.getWindowForTarget")
                    cdp.send("Browser.setWindowBounds", {
                        "windowId": window_info["windowId"],
                        "bounds": {"windowState": "maximized"}
                    })
                except Exception as e:
                    print(f"Failed to maximize window using CDP: {e}")

                # Navigate to WhatsApp Web if not already there
                if not self.page.url or "web.whatsapp.com" not in self.page.url:
                    self.page.goto("https://web.whatsapp.com", wait_until="domcontentloaded")
            return True
        except Exception as e:
            print(f"Failed to initialize browser: {e}")
            return False

    def _is_qr_code_page(self) -> bool:
        """Check if currently on QR code page."""
        try:

            has_scan_text = self.page.locator("text=scan the QR code").is_visible()
            has_login_text = self.page.locator("text='Log into WhatsApp Web'").is_visible()
            return has_scan_text and has_login_text
        except:
            return False

    def _is_loading_messages(self) -> Optional[int]:
        """
        Check if messages are loading and return percentage if found.
        Returns: Percentage loaded if on loading page, None otherwise.
        """
        try:
            loading_text = self.page.locator("text=Loading your chats").first.text_content()
            if "Don't close this window" in self.page.content():
                # Extract percentage from "Loading your chats [xx%]"
                if match := re.search(r"\[(\d+)%\]", loading_text):
                    return int(match.group(1))
        except:
            pass
        return None

    def _calculate_wait_time(self, current_percentage: int, last_percentage: int, last_wait_time: float) -> float:
        """Calculate wait time based on loading progress."""
        if current_percentage <= last_percentage:
            # If progress is stuck, increase wait time
            return min(last_wait_time * 1.5, 10.0)  # Cap at 10 seconds
        else:
            # Progress is being made, adjust wait time based on speed
            progress_rate = current_percentage - last_percentage
            if progress_rate > 20:
                return 1.0  # Fast progress, check quickly
            elif progress_rate > 10:
                return 2.0  # Moderate progress
            else:
                return 3.0  # Slow progress

    def _is_end_to_end_encrypted(self) -> bool:
        """Check if end-to-end encryption notice is visible."""
        try:
            return "End-to-end encrypted" in self.page.content()
        except:
            return False

    def _validate_storage_state(self, storage_state: dict) -> bool:
        """Validate storage state contains required WhatsApp Web data."""
        try:
            # Basic structure check
            if not isinstance(storage_state, dict):
                print("Storage state is not a dictionary")
                return False

            # Check cookies
            cookies = storage_state.get("cookies", [])
            if not isinstance(cookies, list):
                print("Cookies is not a list")
                return False

            # Verify essential WhatsApp cookies
            wa_cookies = {
                cookie["name"]: cookie for cookie in cookies if cookie.get("domain", "").endswith("web.whatsapp.com")
            }
            if "wa_ul" not in wa_cookies:
                print("Missing essential WhatsApp cookie: wa_ul")
                return False

            # Check origins
            origins = storage_state.get("origins", [])
            if not isinstance(origins, list):
                print("Origins is not a list")
                return False

            # Look for WhatsApp Web origin
            for origin in origins:
                if origin.get("origin") == "https://web.whatsapp.com":
                    return True

            print("No WhatsApp Web origin found")
            return False
        except Exception as e:
            print(f"Storage state validation failed: {e}")
            return False

    def wait_for_login(self, timeout: int = 15 * 60) -> bool:
        """Wait for WhatsApp Web login to complete."""
        try:
            start_time = time.time()
            last_percentage = 0
            wait_time = 1
            retry_num = 0

            while time.time() - start_time < timeout:
                # Check if already on chat list
                if self.page.locator('h1:has-text("Chats")').is_visible(timeout=1 * 60 * 1000):
                    print("Chat list is visible!")

                    # Wait for service worker and IndexedDB to be ready
                    time.sleep(3)  # Give time for IndexedDB to initialize

                    if self.keep_open:
                        print("Browser will remain open")
                    return True

                # Check for QR code page
                if self._is_qr_code_page():
                    print("Please scan the QR code with your phone...")
                    input("Press Enter after scanning the QR code...")
                    continue

                # Check for loading page
                if current_percentage := self._is_loading_messages():
                    print(f"Loading messages: {current_percentage}%")
                    wait_time = self._calculate_wait_time(current_percentage, last_percentage, wait_time)
                    last_percentage = current_percentage
                    time.sleep(wait_time)
                    continue

                # Check for end-to-end encrypted message
                if self._is_end_to_end_encrypted():
                    print("Waiting for chats to load...")
                    time.sleep(2)
                    continue

                # If none of the above cases match, we're on an unrecognized page
                if retry_num > 3:
                    print("Unrecognized Page, script is halted")
                    return False

                retry_num += 1
                time.sleep(1)

            print("Login timed out!")
            return False
        except Exception as e:
            print(f"Login failed: {e}")
            return False

    def navigate_to_archived(self) -> bool:
        """Navigate to the archived chats page."""
        # First check if we're already on the archived page
        if self.page.locator(SELECTORS["ARCHIVED_HEADER"]).is_visible(timeout=1000):
            print("Already on archived chats page")
            return True

        # Wait for the chat list and click the Archived link
        self.page.wait_for_selector('h1:has-text("Chats")', timeout=10000)
        archived_link = self.page.locator(SELECTORS["ARCHIVED_TEXT"]).first

        if not archived_link.is_visible():
            print("Could not find Archived link")
            return False

        archived_link.click()

        # Wait for archived header to confirm we're on archived page
        self.page.wait_for_selector(SELECTORS["ARCHIVED_HEADER"], timeout=5000)
        print("Successfully navigated to archived chats")
        return True

    def get_image_base64(self, page, blob_url: str) -> str:
        """Get base64 string of image from blob URL synchronously using fetch and ArrayBuffer conversion."""
        try:
            js_code = """
            async (url) => {
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                // Read raw bytes from the response
                const buffer = await response.arrayBuffer();
                // Convert the array buffer to a binary string
                let binary = '';
                const bytes = new Uint8Array(buffer);
                for (let i = 0; i < bytes.byteLength; i++) {
                    binary += String.fromCharCode(bytes[i]);
                }
                // Convert binary string to Base64
                const base64String = btoa(binary);
                const contentType = response.headers.get('content-type') || 'image/jpeg';
                const dataUrl = `data:${contentType};base64,` + base64String;
                return dataUrl;
            }
            """
            result = page.evaluate(js_code, blob_url)
            # Check for known placeholder signature (1x1 transparent GIF)
            if result.startswith("data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP"):
                raise Exception("Returned image appears to be a placeholder")
            return result
        except Exception as e:
            print(f"Failed to get image base64: {e}")
            return None

    def parse_messages(self, html: str, start_datetime: datetime, group_name: str) -> Dict[str, Dict]:
        """Parse messages from HTML content after start_datetime."""
        soup = BeautifulSoup(html, "html.parser")
        messages = {}
        current_date = None
        event_pattern = "|".join(EVENT_PATTERNS)

        chat_container = soup.find("div", {"role": "application"})
        if not chat_container:
            return messages

        for div in chat_container.find_all("div", recursive=False):
            # Handle date headers
            if "focusable-list-item" in div.get("class", []):
                date_span = div.find("span", dir="auto")
                if date_span and date_span.text:
                    current_date = convert_whatsapp_date(date_span.text.strip())
                    continue

            if div.get("role") == "row" and current_date and current_date >= start_datetime.date():
                try:
                    # Try to get the copyable text block
                    copyable_text = div.find("div", class_="copyable-text")
                    message_text = ""
                    time_str = None

                    if copyable_text:
                        # Extract timestamp from data-pre-plain-text attribute
                        pre_text = copyable_text.get("data-pre-plain-text", "")
                        time_match = re.search(r"\[(.*?),.*?\]", pre_text)
                        if time_match:
                            time_str = time_match.group(1).strip()

                        # Extract text content if available
                        message_content = copyable_text.find("span", class_="selectable-text")
                        if message_content:
                            for elem in message_content.contents:
                                if isinstance(elem, str):
                                    message_text += elem
                                elif elem.name == "br":
                                    message_text += "\n"
                                else:
                                    message_text += elem.get_text()
                            message_text = message_text.strip()
                    else:
                        # For image-only messages, try to extract timestamp from a span matching a time pattern.
                        timestamp_span = div.find("span", text=re.compile(r"\d{1,2}:\d{2}\s*(?:am|pm)", re.IGNORECASE))
                        if timestamp_span:
                            time_str = timestamp_span.get_text(strip=True)

                    # If no timestamp is found, skip this message.
                    if not time_str:
                        continue

                    # Parse the time string (assumes format like "7:20 am")
                    try:
                        message_time = datetime.strptime(time_str, "%I:%M %p").time()
                    except Exception:
                        continue

                    message_datetime = datetime.combine(current_date, message_time)
                    message_datetime = message_datetime.replace(tzinfo=ZoneInfo("Asia/Kolkata"))

                    # Extract images from the row
                    images = []
                    img_tags = div.find_all("img", attrs={"draggable": True, "style": True, "src": True, "tabindex": False})
                    for img in img_tags:
                        blob_url = img.get("src")
                        if not blob_url:
                            continue

                        # Screening: skip known placeholder blobs (e.g., 1x1 transparent GIF)
                        if "R0lGODlhAQABAIAAAAAAAP" in blob_url:
                            continue

                        # Screening: check inline style for extremely small dimensions
                        style = img.get("style", "").lower()
                        if "width:1px" in style or "height:1px" in style:
                            continue

                        try:
                            base64_data = self.get_image_base64(self.page, blob_url)
                            if base64_data and "R0lGODlhAQABAIAAAAAAAP" not in base64_data:
                                images.append(base64_data)
                        except Exception as e:
                            print(f"Failed to get image data: {e}")

                    # Only skip the message if both text and images are empty.
                    if not message_text and not images:
                        continue

                    is_chatty_group = GROUPS.get(group_name, {}).get("chatter", False)
                    if not is_chatty_group or (is_chatty_group and re.search(event_pattern, message_text.lower())):
                        key = message_datetime.strftime("%Y-%m-%dT%H:%M%z")
                        if key in messages:
                            messages[key].append({
                                "text": message_text,
                                "imgs": images,
                            })
                        else:
                            messages[key] = [{
                                "text": message_text,
                                "imgs": images,
                            }]
                except Exception as e:
                    print(f"Error parsing message: {e}")
                    continue

        return messages

    def _wait_for_sync_completion(self, max_wait: int = 120, timeout: int = 900) -> bool:
        """Wait for sync message to disappear with exponential backoff.

        Args:
            max_wait: Maximum wait time between checks in seconds (default 120)
            timeout: Total timeout in seconds (default 600)
        Returns:
            bool: True if sync completed, False if timed out
        """
        base_wait = 1
        current_wait = base_wait
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                sync_message = self.page.locator("div:has-text('Syncing older messages. Click to see progress.')").first
                if not sync_message.is_visible(timeout=1000):
                    return True

                print(f"Waiting for message sync... (next check in {current_wait}s)")
                time.sleep(current_wait)

                # Exponential backoff with max cap
                current_wait = min(current_wait * 2, max_wait)

            except Exception as e:
                print(f"Error checking sync status: {e}")
                return True  # Assume sync completed if we can't find the message

        print(f"Sync wait timed out after {timeout} seconds")
        return False

    def scrape_group(
        self, group_name: str, days_back_to_process: int = 10, verbose: bool = False
    ) -> Optional[Dict[str, str]]:
        """Scrape messages from a WhatsApp group.

        Args:
            group_name: Name of the group to scrape
            days_back_to_process: Number of days to look back for messages
            verbose: Whether to show sample first and last messages

        Returns:
            Optional[Dict[str, str]]: Messages with datetime as key and text as value, or None if no messages found
        """
        print(f"Scraping group: {group_name}")

        # Wait for sync to complete after selecting chat
        if not self._wait_for_sync_completion():
            print("Warning: Message sync timed out")

        # Get target date from days_back_to_process
        target_date = datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(days=days_back_to_process)

        # Get last scrape datetime
        last_scrape_dt = self.messages_handler.get_last_scrape_datetime()

        # Use whichever is more recent
        start_datetime_to_scrape = max(target_date, last_scrape_dt) if last_scrape_dt else target_date
        print(f"Looking for messages after: {start_datetime_to_scrape}")

        # Look for scroll to bottom button and click if present
        scroll_button = self.page.locator(SELECTORS["CHAT_SCROLL_BOTTOM_BUTTON"]).first
        if scroll_button.is_visible():
            print("Found scroll to bottom button, clicking...")
            scroll_button.click()
            time.sleep(3)  # Wait for scrolling to complete

        # Get the chat scroll container
        scroll_container = self.page.locator(SELECTORS["CHAT_SCROLL_CONTAINER"]).first

        def check_target_date():
            """Check if target date is in current view."""
            date_divs = self.page.locator(SELECTORS["MESSAGE_DATE_DIVS"]).all()
            oldest_date = None

            for div in date_divs:
                try:
                    if div.is_visible():
                        date_text = div.text_content().strip()
                        parsed_date = convert_whatsapp_date(date_text)
                        if parsed_date:
                            if oldest_date is None or parsed_date < oldest_date:
                                oldest_date = parsed_date
                                if parsed_date <= start_datetime_to_scrape.date():
                                    print(f"Found target date: {parsed_date}")
                                    return True
                except Exception as e:
                    print(f"Failed to parse date: {e}")

            return False

        # Scroll up until target date is found
        target_found = False
        no_scroll_count = 0
        while not target_found:
            # Check current view for target date
            target_found = check_target_date()
            if target_found:
                break

            # Check for older messages button
            older_messages = self.page.locator(SELECTORS["OLDER_MESSAGES_BUTTON"]).first
            if older_messages.is_visible():
                print("Loading older messages...")
                older_messages.click()
                time.sleep(0.5)
                continue

            # Scroll to top
            current_scroll = scroll_container.evaluate("el => el.scrollTop")
            scroll_container.evaluate("el => el.scrollTop = 0")
            new_scroll = scroll_container.evaluate("el => el.scrollTop")

            if current_scroll == new_scroll:
                no_scroll_count += 1
                if no_scroll_count > 3:
                    print("Reached the top of the chat")
                    break

            time.sleep(2)  # Wait for content to load

        if target_found:
            # Handle all "Read more" buttons
            print("Expanding all 'Read more' messages...")
            while True:
                read_more_buttons = self.page.locator(SELECTORS["READ_MORE_BUTTON"]).all()
                if not read_more_buttons:
                    break

                clicked = False
                for button in read_more_buttons:
                    try:
                        if button.is_visible():
                            button.scroll_into_view_if_needed(timeout=5000)
                            time.sleep(0.5)

                            if button.is_visible():
                                button.click()
                                clicked = True
                                time.sleep(0.5)  # Wait for expansion
                    except Exception as e:
                        print(f"Failed to click read more button: {e}")

                if not clicked:
                    break

            # Parse messages
            print("Parsing messages...")
            html = self.page.content()
            messages = self.parse_messages(html, start_datetime_to_scrape, group_name)

            if messages:
                print(f"{group_name} completed: {len(messages)} messages scraped")
                if verbose:
                    sorted_timestamps = sorted(messages.keys())
                    print(f"First message ({sorted_timestamps[0]}):\n{messages[sorted_timestamps[0]][:200]}...")
                    print(f"Last message ({sorted_timestamps[-1]}):\n{messages[sorted_timestamps[-1]][:200]}...")
                return messages

            print(f"{group_name} completed: 0 messages scraped")
            return None

    def find_groups(self, groups_list: List[str], days_back: int = 7) -> set:
        """Find specified groups in the archived chats.

        Args:
            groups_list: List of group names to search for
            days_back: Number of days to look back for messages

        Returns:
            Set of found group names
        """
        # If test_run is True, randomly select 3 groups
        if self.test_run:
            groups_list = ["Bangalore IRLs"]

        if not self.navigate_to_archived():
            print("Could not access archived chats")
            return set()

        scroll_container = self.page.locator(SELECTORS["SCROLL_CONTAINER"]).first

        # Scroll to top first
        print("Scrolling to top of archived chats...")
        scroll_container.evaluate("el => el.scrollTop = 0")
        time.sleep(2)  # Wait for content to load

        found_groups = set()
        remaining_groups = set(groups_list)  # Track unfound groups
        groups_data = []  # Store all group data for messages.json
        last_scroll_position = 0
        start_time = time.time()
        timeout = 30 * 60  # 30 minutes timeout

        # Get container height and calculate fixed scroll amount (70%)
        container_height = scroll_container.evaluate("el => el.clientHeight")
        scroll_amount = int(container_height * 0.7)

        while len(found_groups) < len(groups_list):
            if time.time() - start_time > timeout:
                print("Stopped due to timeout (30 minutes)")
                break

            # Process visible groups
            group_elements = self.page.locator(SELECTORS["ARCHIVED_GROUPS"]).all()

            # Check each visible group
            for element in group_elements:
                group_name = element.text_content()
                # Check if any name in remaining_groups is contained within this group name
                matching_groups = [g for g in remaining_groups if g.lower() in group_name.lower()]
                if matching_groups:
                    if len(matching_groups) > 1:
                        raise Exception(
                            f"Error: Imprecise group name in group list for {group_name}. Multiple matches: {matching_groups}"
                        )

                    matched_group = matching_groups[0]
                    if group_name not in found_groups:
                        print(f"Found group: {group_name} (matched: {matched_group})")
                        element.click()
                        time.sleep(2)  # Wait for group to load

                        # Wait for sync to complete after clicking group
                        if not self._wait_for_sync_completion():
                            print(f"Warning: Message sync timed out for {matched_group}")
                            continue

                        # Verify announcement group if needed
                        if not self._verify_announcement_group(matched_group):
                            print(f"Skipping {matched_group}: Not the announcement group we're looking for")
                            continue

                        if messages := self.scrape_group(matched_group, days_back):
                            # Add group data with messages
                            group_data = {
                                "group_name": matched_group,
                                "created_at": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%dT%H:%M%z"),
                                "messages": messages,
                            }
                            groups_data.append(group_data)

                        found_groups.add(group_name)
                        remaining_groups.remove(matched_group)

            # Calculate new scroll position
            new_scroll_position = last_scroll_position + scroll_amount
            current_scroll_height = scroll_container.evaluate("el => el.scrollHeight")

            # Check if we've reached the bottom
            if new_scroll_position >= current_scroll_height or new_scroll_position <= last_scroll_position:
                print("Reached end of archived chats")
                break

            # Perform the scroll
            scroll_container.evaluate(f"el => el.scrollTop = {new_scroll_position}")
            current_position = scroll_container.evaluate("el => el.scrollTop")

            if current_position == last_scroll_position:
                print("Reached end of archived chats")
                break

            last_scroll_position = current_position

            # Wait for content to load
            time.sleep(1)

        # After processing all groups, create the messages file
        if groups_data:
            self.messages_handler.create_messages_file(groups_data)
            print(f"Created messages file with {len(groups_data)} groups")

        print(f"Found {len(found_groups)} out of {len(groups_list)} groups")
        if remaining_groups:
            print(f"Groups not found: {remaining_groups}")
        return found_groups
```
-----

FILE: whatsapp_day_to_date.py
CONTENT:
```py
#%%
from datetime import datetime, timedelta
from typing import Dict, Optional

def get_whatsapp_date_mapping() -> Dict[str, datetime.date]:
	"""
	Creates a mapping between WhatsApp day references and actual dates.

	Returns:
		Dict[str, datetime.date]: Mapping of WhatsApp day references to actual dates
	"""
	today = datetime.today()
	mapping = {
		"TODAY": today.date(),
		"YESTERDAY": (today - timedelta(days=1)).date()
	}

	# Map weekday names
	weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
	for i, day in enumerate(weekdays):
		if today.weekday() < i:
			mapping[day] = today.date() + timedelta(days=i - today.weekday() - 7)
		else:
			mapping[day] = today.date() + timedelta(days=i - today.weekday())

	return mapping

def convert_whatsapp_date(date_text: str) -> Optional[datetime.date]:
	"""
	Converts WhatsApp date text to actual date object.

	Args:
		date_text (str): Date text from WhatsApp (e.g., "TODAY", "YESTERDAY", "DD/MM/YYYY")

	Returns:
		Optional[datetime.date]: Converted date or None if conversion fails
	"""
	try:
		date_text = date_text.strip().upper()
		mapping = get_whatsapp_date_mapping()

		if date_text in mapping:
			return mapping[date_text]

		# Try parsing as DD/MM/YYYY"
		return datetime.strptime(date_text, "%d/%m/%Y").date()
	except (ValueError, AttributeError):
		return None
```
