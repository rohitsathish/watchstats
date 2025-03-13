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


# Guidelines
- Always make a plan and execute step by step.
- Propose and implement the most suitable approach - the one that optimally balances low code complexity and high performance/low latency
- Always write creating clean, simple, efficient code that effectively fulfiills the user request.

## Backend - Python

### Code Quality
- Modularize code into files based on functionality.
- Write concise Google-style docstrings.
- Load environment variables via `.env` in a dedicated config.
- Implement try-except blocks for common issues

### User Preferences
- Use Plotly for plotting.
- Use Polars for data processing, leveraging lazy loading, vectorized operations, and multi-threading for optimal performance.

---

## Frontend - Vite, React and Chakra UI

### Project Setup
- Keep dependencies minimal and optimize performance with tree-shaking and lazy loading.

### Code Structure
- Modular, scalable and reusable components.
- Keep components modular (/components), organize pages (/pages), and configure Chakra UI in a central theme file (`theme.js`). 
- Use React Query (TanStack Query) for state management
- Use Chakra UI’s component-based styling to avoid unnecessary CSS complexity.  

### Performance
- Prioritize performance via Vite features (fast HMR, minification, lazy loading).
- Prioritize user experience and development efficiency.

