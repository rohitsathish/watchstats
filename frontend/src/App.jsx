import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Container, VStack } from '@chakra-ui/react';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import WatchHistoryPage from './pages/WatchHistoryPage';
import AuthCallback from './pages/AuthCallback';
import ConnectionStatus from './components/ConnectionStatus';
import { useAuth } from './context/AuthContext';
import { WatchHistoryProvider } from './context/WatchHistoryContext';

//#%% Main App Component
function App() {
  const [apiStatus, setApiStatus] = useState('Unknown');
  const { connectionStatus } = useAuth();

  // Update API status based on socket connection
  useEffect(() => {
    if (connectionStatus.connected) {
      setApiStatus('Connected');
    } else if (connectionStatus.connecting) {
      setApiStatus('Connecting...');
    } else if (!connectionStatus.connected && connectionStatus.error) {
      setApiStatus('Error');
    } else {
      setApiStatus('Unknown');
    }
  }, [connectionStatus]);

  return (
    <Router>
      <WatchHistoryProvider>
        <VStack minH="100vh" spacing={0}>
          <Header />
          
          <Container maxW="container.xl" py={8} flex="1">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/watch-history" element={<WatchHistoryPage />} />
              <Route path="/auth/callback" element={<AuthCallback />} />
            </Routes>
          </Container>
          
          <Footer apiStatus={apiStatus} />
          <ConnectionStatus />
        </VStack>
      </WatchHistoryProvider>
    </Router>
  );
}

export default App;
