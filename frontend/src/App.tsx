import { useState, useRef, useEffect } from 'react';
import { useAuth, UserButton, SignInButton } from '@clerk/clerk-react';
import { 
  Button, 
  Container, 
  Box, 
  Typography, 
  Alert, 
  Card, 
  CardContent, 
  CircularProgress, 
  AppBar, 
  Toolbar, 
  Tabs, 
  Tab
} from '@mui/material';
import { 
  Upload as UploadIcon,
  Home as HomeIcon,
  Face as FaceIcon 
} from '@mui/icons-material';
// Removed unused imports
import { BrowserRouter as Router, Routes, Route, Link, useLocation, useNavigate } from 'react-router-dom';
import { analyzeVideo } from './apiService';
import config from './config';
import FaceIndex from './components/FaceIndex';

interface DetectionResult {
  is_deepfake: boolean;
  confidence: number;
  processed_at?: string;
  processed_frames?: number;
  frame_count?: number;
  details?: any;
}

function Navigation() {
  const location = useLocation();
  const navigate = useNavigate();
  const { isLoaded, isSignedIn } = useAuth();
  const [value, setValue] = useState(location.pathname === '/face-index' ? 1 : 0);

  const handleChange = (_: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
    navigate(newValue === 1 ? '/face-index' : '/');
  };

  if (!isLoaded) {
    return null; // or a loading spinner
  }

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Deepfake Detection
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Tabs 
            value={value} 
            onChange={handleChange}
            textColor="inherit"
            indicatorColor="secondary"
          >
            <Tab label="Home" icon={<HomeIcon />} component={Link} to="/" />
            {isSignedIn && (
              <Tab label="Face Index" icon={<FaceIcon />} component={Link} to="/face-index" />
            )}
          </Tabs>
          <Box sx={{ ml: 'auto' }}>
            {isSignedIn ? (
              <UserButton afterSignOutUrl="/" />
            ) : (
              <SignInButton mode="modal">
                <Button color="inherit">Sign In</Button>
              </SignInButton>
            )}
          </Box>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [healthStatus, setHealthStatus] = useState<'healthy' | 'unhealthy' | 'checking'>('checking');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check backend health on component mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/api/health`);
        if (response.ok) {
          setHealthStatus('healthy');
          setError(null);
        } else {
          throw new Error('Health check failed');
        }
      } catch (err: any) {
        console.error('Health check failed:', err);
        setHealthStatus('unhealthy');
        setError('Backend is not reachable. Please ensure the backend server is running.');
      }
    };

    checkBackendHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await analyzeVideo(file);
      setResult(response);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Error processing video. Please try again.';
      setError(errorMessage);
      console.error('Video analysis failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Deepfake Detection
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Upload a video to check for deepfake content
        </Typography>
        <Typography variant="body2" color={healthStatus === 'healthy' ? 'success.main' : 'error.main'}>
          Backend Status: {healthStatus}
        </Typography>
      </Box>

      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Box
            component="form"
            onSubmit={handleSubmit}
            sx={{
              border: '2px dashed #ccc',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              '&:hover': {
                borderColor: 'primary.main',
                cursor: 'pointer',
              },
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="video/*"
              style={{ display: 'none' }}
              disabled={isLoading}
            />
            
            <UploadIcon sx={{ fontSize: 40, mb: 2, color: 'text.secondary' }} />
            <Typography variant="h6" gutterBottom>
              {file ? file.name : 'Click to select a video file'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {file ? 'Click to change file' : 'or drag and drop file here'}
            </Typography>
            
            {file && !isLoading && (
              <Button
                type="submit"
                variant="contained"
                color="primary"
                onClick={(e) => e.stopPropagation()}
                sx={{ mt: 2 }}
              >
                Analyze Video
              </Button>
            )}

            {isLoading && (
              <Box sx={{ width: '100%', mt: 2 }}>
                <CircularProgress />
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Processing video...
                </Typography>
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </Box>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Analysis Results
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Typography>
                <strong>Status:</strong>{' '}
                <span style={{ color: result.is_deepfake ? 'red' : 'green' }}>
                  {result.is_deepfake ? 'Deepfake Detected' : 'Authentic'}
                </span>
              </Typography>
              <Typography>
                <strong>Confidence:</strong> {Math.round(result.confidence * 100)}%
              </Typography>
              <Typography>
                <strong>Processed At:</strong>{' '}
                {new Date(result.details?.processed_at).toLocaleString()}
              </Typography>
              <Typography>
                <strong>Frames Processed:</strong> {result.details?.processed_frames} of {result.details?.frame_count}
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}
    </Container>
  );
}

function App() {
  return (
    <Router>
      <Box sx={{ flexGrow: 1 }}>
        <Navigation />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/face-index" element={<FaceIndex />} />
        </Routes>
      </Box>
    </Router>
  );
}

export default App;
