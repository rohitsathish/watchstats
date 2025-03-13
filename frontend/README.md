# Trakt Media Visualizer - Frontend

This is a React + Vite + ChakraUI frontend for the Trakt Media Visualizer API. The application allows users to login with their Trakt.tv account and view statistics about their watched movies and TV shows.

## Features

- **OAuth Authentication**: Login with your Trakt.tv account
- **Watch Statistics**: View summaries of your watch history
- **Top Genres**: See your most-watched genres
- **Time Filtering**: Filter statistics by time period (last month, 3 months, 6 months, 1 year, all time)
- **Responsive Design**: Works on desktop and mobile devices
- **PWA Support**: Can be installed as a Progressive Web App

## Technologies Used

- **React**: JavaScript library for building user interfaces
- **Vite**: Next-generation frontend tooling
- **ChakraUI**: Component library for building accessible and themeable React applications
- **React Router**: Routing library for React
- **Axios**: Promise-based HTTP client

## Getting Started

1. Make sure the FastAPI backend is running at http://localhost:8000
2. Install dependencies:
   ```
   npm install
   ```
3. Start the development server:
   ```
   npm run dev
   ```
4. Open your browser and navigate to http://localhost:3000

## Building for Production

To create a production build:

```
npm run build
```

The build artifacts will be stored in the `dist/` directory.

## API Integration

This frontend integrates with the Trakt Media Visualizer API to:

- Authenticate users with Trakt.tv
- Fetch watch statistics
- Get genre information
- Display media details

## Authentication Flow

1. User clicks "Login with Trakt"
2. They are redirected to Trakt.tv to authorize the app
3. After authorization, Trakt redirects back to the app with an authorization code
4. The app exchanges this code for an access token via the backend API
5. The token is stored in localStorage and used for subsequent API requests
