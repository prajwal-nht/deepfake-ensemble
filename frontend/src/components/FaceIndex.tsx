import { useState, useRef } from 'react';
import { 
  Button, 
  Container, 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  List, 
  ListItem, 
  ListItemText, 
  IconButton, 
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  DialogContentText,
  Alert
} from '@mui/material';
import { Upload as UploadIcon, Delete as DeleteIcon, Face as FaceIcon } from '@mui/icons-material';
import { useAuth, SignInButton } from '@clerk/clerk-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { faceApi } from '../apiService';

interface FaceIndexType {
  id: string;
  user_id: string;
  name: string;
  created_at: string;
  vector_embedding?: number[];
}

export default function FaceIndex() {
  const { userId, isLoaded } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [faceToDelete, setFaceToDelete] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  // Debug log
  console.log('FaceIndex component mounted with userId:', userId);

  // Fetch face indices
  const { data: faceIndices = [], isLoading: isLoadingIndices, error } = useQuery<FaceIndexType[]>({
    queryKey: ['faceIndices', userId],
    queryFn: async () => {
      console.log('Fetching face indices for user:', userId);
      try {
        const data = await faceApi.getFaceIndices(userId!);
        console.log('Received face indices:', data);
        return data;
      } catch (err) {
        console.error('Error fetching face indices:', err);
        throw err;
      }
    },
    enabled: !!userId && isLoaded,
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: (file: File) => {
      if (!userId) throw new Error('User not authenticated');
      return faceApi.indexFace(userId, file);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['faceIndices'] });
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (faceId: string) => faceApi.deleteFaceIndex(faceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['faceIndices'] });
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleIndexFace = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !userId) return;
    uploadMutation.mutate(file);
  };

  const handleDeleteFaceIndex = (faceId: string) => {
    setFaceToDelete(faceId);
    setDeleteDialogOpen(true);
  };

  const handleConfirmDelete = () => {
    if (faceToDelete) {
      deleteMutation.mutate(faceToDelete);
    }
    setDeleteDialogOpen(false);
  };

  const handleCancelDelete = () => {
    setDeleteDialogOpen(false);
    setFaceToDelete(null);
  };

  // Show loading state while checking auth
  if (!isLoaded) {
    return (
      <Container maxWidth="md" sx={{ py: 4, textAlign: 'center' }}>
        <CircularProgress />
        <Typography>Loading authentication...</Typography>
      </Container>
    );
  }

  // Handle unauthenticated state
  if (!userId) {
    return (
      <Container maxWidth="md" sx={{ py: 4, textAlign: 'center' }}>
        <Typography variant="h5" gutterBottom>
          Authentication Required
        </Typography>
        <Typography sx={{ mb: 2 }}>
          Please sign in to access the Face Index feature.
        </Typography>
        <SignInButton mode="modal">
          <Button variant="contained" color="primary">
            Sign In
          </Button>
        </SignInButton>
      </Container>
    );
  }

  // Delete confirmation dialog
  const DeleteDialog = () => (
    <Dialog open={deleteDialogOpen} onClose={handleCancelDelete}>
      <DialogTitle>Delete Face Index</DialogTitle>
      <DialogContent>
        <DialogContentText>
          Are you sure you want to delete this face index? This action cannot be undone.
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleCancelDelete}>Cancel</Button>
        <Button onClick={handleConfirmDelete} color="error">
          Delete
        </Button>
      </DialogActions>
    </Dialog>
  );

  // Test if component renders
  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" color="primary" gutterBottom>
        Face Index Test
      </Typography>
      <Typography>User ID: {userId || 'Not logged in'}</Typography>
      <Typography>Is Loaded: {isLoaded.toString()}</Typography>
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          Error: {error.message}
        </Alert>
      )}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Face Indexing
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Upload face images to create a reference for face recognition
        </Typography>
      </Box>

      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Box
            component="form"
            onSubmit={handleIndexFace}
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
              accept="image/*"
              style={{ display: 'none' }}
              disabled={uploadMutation.isPending}
            />
            
            {uploadMutation.isPending ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <CircularProgress size={24} sx={{ mb: 1 }} />
                <Typography>Indexing face...</Typography>
              </Box>
            ) : (
              <>
                <UploadIcon fontSize="large" sx={{ mb: 1 }} />
                <Typography variant="h6" sx={{ wordBreak: 'break-word' }}>
                  {file?.name || 'Click to select an image'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {file ? 'Click to upload' : 'or drag and drop'}
                </Typography>
              </>
            )}
          </Box>
          
          {file && !uploadMutation.isPending && (
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleIndexFace}
                disabled={!file || !userId}
                fullWidth
              >
                Index Face
              </Button>
            </Box>
          )}

          {uploadMutation.isError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {String(uploadMutation.error)}
            </Alert>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Your Face Indices
          </Typography>
          
          {isLoadingIndices ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress />
            </Box>
          ) : faceIndices.length > 0 ? (
            <List>
              {faceIndices.map((faceIndex) => (
                <ListItem
                  key={faceIndex.id}
                  secondaryAction={
                    <IconButton 
                      edge="end" 
                      aria-label="delete"
                      onClick={() => handleDeleteFaceIndex(faceIndex.id)}
                      disabled={deleteMutation.isPending}
                    >
                      {deleteMutation.variables === faceIndex.id && deleteMutation.isPending ? (
                        <CircularProgress size={24} />
                      ) : (
                        <DeleteIcon />
                      )}
                    </IconButton>
                  }
                >
                  <FaceIcon sx={{ mr: 2, color: 'primary.main' }} />
                  <ListItemText
                    primary={faceIndex.name}
                    secondary={
                      <>
                        <Box component="span" display="block">
                          Created: {new Date(faceIndex.created_at).toLocaleString()}
                        </Box>
                        <Box component="span" display="block">
                          ID: {faceIndex.id}
                        </Box>
                        <Box component="span" display="block">
                          Embeddings: {faceIndex.vector_embedding ? 'Available' : 'None'}
                        </Box>
                      </>
                    }
                  />
                </ListItem>
              ))}
            </List>
          ) : (
            <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
              No face indices found. Upload an image to get started.
            </Typography>
          )}

          {deleteMutation.isError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              Error deleting face index: {String(deleteMutation.error)}
            </Alert>
          )}
        </CardContent>
      </Card>

      <DeleteDialog />
    </Container>
  );
}