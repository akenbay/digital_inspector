import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [processedImage, setProcessedImage] = useState(null);
  const [detections, setDetections] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setProcessedImage(null);
      setDetections(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setDetections(response.data.detections);
      
      // Load the processed image
      const imageUrl = `${API_URL}${response.data.download_url}`;
      setProcessedImage(imageUrl);

    } catch (err) {
      setError(
        err.response?.data?.detail || 
        'An error occurred while processing the document'
      );
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  const handleDownload = () => {
    if (processedImage) {
      const link = document.createElement('a');
      link.href = processedImage;
      link.download = `processed_${file.name}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <h1 className="title">Document Processor</h1>
        <p className="subtitle">Upload a document and detect elements using YOLO</p>

        <div className="upload-section">
          <div className="file-input-wrapper">
            <input
              type="file"
              id="file-upload"
              className="file-input"
              accept="image/*,.pdf,.png,.jpg,.jpeg"
              onChange={handleFileChange}
            />
            <label htmlFor="file-upload" className="file-label">
              {file ? file.name : 'Choose a file'}
            </label>
          </div>

          <button
            className="upload-button"
            onClick={handleUpload}
            disabled={!file || uploading}
          >
            {uploading ? 'Processing...' : 'Process Document'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {detections !== null && (
          <div className="info-box">
            <strong>Detections found:</strong> {detections}
          </div>
        )}

        {processedImage && (
          <div className="result-section">
            <h2 className="result-title">Processed Document</h2>
            <div className="image-container">
              <img
                src={processedImage}
                alt="Processed document"
                className="processed-image"
              />
            </div>
            <button className="download-button" onClick={handleDownload}>
              Download Processed Image
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

