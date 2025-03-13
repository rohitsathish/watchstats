# Project Structure

## Overview
A media visualization platform with:
- Backend: FastAPI (in `/app`)
- Frontend: React with Vite and Chakra UI (in `/frontend`)

---

## Backend (`/app`)
FastAPI backend serving APIs and Server-Sent Events (SSE) for real-time updates.

### Main Files
- `main.py`: Initializes the FastAPI app, middleware (CORS, session, auth), defines lifecycle events.

### Subdirectories

#### 📁 `api`
Endpoint definitions, request handling and route-specific logic.
- `__init__.py`: Imports API endpoints.
- `endpoints/`: Contains individual endpoint modules
  - `auth.py`: Manages OAuth login, callbacks, JWT tokens, cookies, and SSE notifications.
  - `events.py`: Handles SSE endpoints for real-time updates.
  - `watch_history.py`: Handles media data endpoints for Trakt.

#### 📁 `core`
Contains foundational utilities, integrations and low-level api interaction.
- `__init__.py`: Imports API endpoints.
- `auth.py`: Authentication logic (OAuth, JWT).
- `config.py`: Loads application configuration settings.
- `sse.py`: Implements Server-Sent Events for real-time communication.
- `trakt_api.py`: Manages api calls to Trakt for media related data.

#### 📁 `db`
Database and cache interaction logic.
- `__init__.py`: Database module initialization.
- `db.py`: Database connections, CRUD operations.
- `schema.sql`: Schema for database initialization.

#### 📁 `services`
Data processing logic.
- `watch_service.py`: Processes and handles Trakt data with polars and caching.

---

## Frontend (`frontend`)
React application built with Vite and Chakra UI interacting with FastAPI backend.

### Root Files
- `vite.config.js`: Vite config (proxy settings, CORS).
- `index.html`: HTML entry point.
- `package.json`: Project dependencies and scripts.

### Main Directories

#### 📁 `public`
Static frontend assets.

#### 📁 `src`
React app source code and structure.

##### Main Files
- `App.css`: Main application styles.
- `App.jsx`: Main application component.
- `index.css`: Base styles.
- `main.jsx`: Root React rendering entry.
- `theme.js`: Chakra UI theme customization.

#### Subdirectories within `src`

##### 📂 `components`
Reusable React components.
(Various components are present)

##### 📂 `context`
Global React contexts.
- `AuthContext.jsx`: Authentication context for user state management.
- `WatchHistoryContext.jsx`: Authentication context for user state management.


##### 📂 `pages`
React page components for specific routes.
- `AuthCallback.jsx`: OAuth authentication callback handling.
- `HomePage.jsx`: Homepage component.
- `WatchHistoryPage.jsx`

##### 📂 `services`
Utility modules for backend interactions.
- `authService.js`: Authentication service.
- `sseService.js`: Server-Sent Events connection management.
- `watchHistory.js`

##### 📂 `utils`
Common helper functions and API utilities.
- `api.js`: Axios API wrapper.