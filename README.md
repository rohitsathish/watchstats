# Trakt Media Visualizer

A web application for visualizing and analyzing your Trakt.tv watching habits with beautiful charts and insightful statistics.

## Project Structure

This project consists of:
- A **FastAPI backend** that interacts with the Trakt.tv API
- A **React frontend** built with Vite and ChakraUI

## Backend Setup

1. Clone the repository
2. Create a `.env` file in the root directory with the following variables:
```
TRAKT_CLIENT_ID=your_trakt_client_id
TRAKT_CLIENT_SECRET=your_trakt_client_secret
TRAKT_REDIRECT_URI=http://localhost:5173/callback
TMDB_API_KEY=your_tmdb_api_key
JWT_SECRET=your_jwt_secret
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the backend:
```bash
python run_api.py
```

The FastAPI server will start on http://localhost:8000.

## Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend/trakt-frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend development server will start on http://localhost:5173.

## Usage

1. Open your browser and navigate to http://localhost:5173
2. Click on "Login with Trakt" to authenticate with your Trakt.tv account
3. After successful authentication, you'll be redirected to the dashboard where you can view your watch statistics

## API Documentation

The FastAPI backend provides the following endpoints:

- `/api/auth/login` - Get a Trakt authorization URL
- `/api/auth/callback` - Exchange authorization code for access token
- `/api/media/stats` - Get watch statistics
- `/api/trakt/history` - Get watch history
- `/api/trakt/movies` - Get watched movies
- `/api/trakt/shows` - Get watched shows

For detailed API documentation, visit http://localhost:8000/docs when the backend is running.

## Development

The project uses:
- Python 3.9+ for the backend
- React 19+ for the frontend
- ChakraUI for styling
- JWT for authentication
- FastAPI for the backend framework

## Features

- **Authentication**: OAuth2 integration with Trakt.tv
- **Watch History**: Retrieve and analyze your Trakt.tv watch history
- **Statistics**: Get detailed statistics about your media consumption
  - Total watch time
  - Genre breakdown
  - Time-based analysis (daily, weekly, monthly patterns)
- **Media Information**: Access detailed information about movies and TV shows
- **TMDB Integration**: Enhanced data through The Movie Database API

## Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **httpx**: Async HTTP client for API requests
- **JWT**: Token-based authentication for API security

## Important Notes

This implementation uses in-memory storage for authentication tokens, which is suitable for development but not production. In a production environment, you would want to implement proper token storage using a database or cache service.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Trakt.tv API](https://trakt.docs.apiary.io/)
- [The Movie Database API](https://developers.themoviedb.org/3)