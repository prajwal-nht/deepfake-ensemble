// src/api/faceApiService.ts
import axios from 'axios';
import config from './config';

const api = axios.create({
  baseURL: config.apiBaseUrl,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
  withCredentials: true,
});

export interface FaceIndex {
  id: string;
  userId: string;
  fileName: string;
  thumbnailUrl?: string;
  embeddingsFound: number;
  framesProcessedForFaces: number;
  totalFramesInVideo: number;
  createdAt: string;
}

export const faceApi = {
  // Index a new face
  async indexFace(userId: string, file: File) {
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('file', file);
    
    const response = await api.post('/index_face', formData);
    return response.data;
  },

  // Get all face indices
  async getFaceIndices() {
    const response = await api.get('/videos/current-user');
    return response.data;
  },

  // Delete face index
  async deleteFaceIndex(faceId: string) {
    const response = await api.delete(`/videos/${faceId}`);
    return response.data;
  }
};