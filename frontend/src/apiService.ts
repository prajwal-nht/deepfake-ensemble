import axios from 'axios';
import config from './config';

console.log('API Base URL:', config.apiBaseUrl); // Debug log

const api = axios.create({
  baseURL: config.apiBaseUrl,
  timeout: 300000, // 5 minutes
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false, // Disable credentials to work with wildcard CORS
});

// Add a request interceptor to include the auth token if available
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export interface DetectionResult {
  video_id: string;
  is_deepfake: boolean;
  confidence: number;
  processed_at: string;
  face_matches?: any[];
  frame_count?: number;
  processed_frames?: number;
  details?: {
    processed_at: string;
    processed_frames: number;
    frame_count: number;
  };
}

export const analyzeVideo = async (file: File): Promise<DetectionResult> => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post<DetectionResult>('/api/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          console.log(`Upload Progress: ${percentCompleted}%`);
        }
      },
      timeout: 300000, // 5 minutes timeout for this specific request
    });
    console.log('Analysis complete:', response.data);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Failed to analyze video');
    }
    throw error;
  }
};

export const checkHealth = async (): Promise<{ status: string }> => {
  try {
    const response = await api.get<{ status: string }>('/api/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw new Error('Backend is not reachable');
  }
};

export const faceApi = {
  indexFace: async (userId: string, file: File) => {
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('file', file);
    
    const response = await api.post('/api/face_indices', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  getFaceIndices: async (userId: string) => {
    const response = await api.get(`/api/face_indices?user_id=${userId}`);
    return response.data;
  },
  
  deleteFaceIndex: async (faceId: string) => {
    const response = await api.delete(`/api/face_indices/${faceId}`);
    return response.data;
  },
  
  processVideo: async (file: File, userId?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (userId) {
      formData.append('user_id', userId);
    }
    
    const response = await api.post('/api/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  checkStatus: async (jobId: string) => {
    const response = await api.get(`/api/status/${jobId}`);
    return response.data;
  }
};

export default {
  analyzeVideo,
  checkHealth,
  faceApi
};

// Add response interceptor for logging
api.interceptors.response.use(
    response => {
      console.log('Response:', response);
      return response;
    },
    error => {
      console.error('API Error:', error);
      return Promise.reject(error);
    }
  );

// Mark the API service setup as completed in the TODO list
// TODO: Mark "Create and configure API service" as completed
