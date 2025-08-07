import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [firstFrameUrl, setFirstFrameUrl] = useState(null);
  const [points, setPoints] = useState([]);
  const [videoFilename, setVideoFilename] = useState(null);
  const [s3Folder, setS3Folder] = useState(null); // Add S3 folder state
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [segmentedImageUrl, setSegmentedImageUrl] = useState(null);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [maskedVideoUrl, setMaskedVideoUrl] = useState(null);
  const [maskedVideoFilename, setMaskedVideoFilename] = useState(null);
  const [isRunningOpenPose, setIsRunningOpenPose] = useState(false);
  const [landmarkedVideoUrl, setLandmarkedVideoUrl] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [imageRef, setImageRef] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage('');
    setFirstFrameUrl(null);
    setPoints([]);
    setCurrentStep(0);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setMessage('Uploading video...');
    setFirstFrameUrl(null);
    setPoints([]);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setMessage('Upload successful. Please select points on the image below.');
        setVideoFilename(data.video_filename);
        setS3Folder(data.s3_folder); // Store S3 folder
        let url = data.first_frame_url;
        if (url && !url.startsWith('http')) {
          url = `${process.env.NEXT_PUBLIC_API_URL}${url}`;
        }
        setFirstFrameUrl(url);
        setCurrentStep(1);
      } else {
        setMessage('Error: ' + (data.error || 'Upload failed'));
      }
    } catch (err) {
      setMessage('Error: ' + err.message);
    }
  };

  const handleImageClick = async (e) => {
    if (!firstFrameUrl) return;
    const rect = e.target.getBoundingClientRect();
    const img = e.target;
    
    const displayWidth = img.offsetWidth;
    const displayHeight = img.offsetHeight;
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;
    
    const displayX = Math.round(e.nativeEvent.clientX - rect.left);
    const displayY = Math.round(e.nativeEvent.clientY - rect.top);
    
    const originalX = Math.round((displayX / displayWidth) * naturalWidth);
    const originalY = Math.round((displayY / displayHeight) * naturalHeight);
    
    const newPoints = [...points, { x: originalX, y: originalY }];
    setPoints(newPoints);
    
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_URL}/coords`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points: newPoints }),
      });
    } catch (err) {
      // Optionally handle error
    }
  };

  const handleSegment = async () => {
    if (!points.length || !videoFilename) return;
    
    setIsSegmenting(true);
    setMessage('Running SAM2 segmentation...');
    
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/segment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          points: points,
          video_filename: videoFilename 
        }),
      });
      
      const data = await res.json();
      
      if (res.ok) {
        setMessage('Segmentation completed. Review the result below.');
        let url = data.segmented_image_url;
        if (url && !url.startsWith('http')) {
          url = `${process.env.NEXT_PUBLIC_API_URL}${url}`;
        }
        setSegmentedImageUrl(url);
        setCurrentStep(2);
      } else {
        setMessage('Error: ' + (data.error || 'Segmentation failed'));
      }
    } catch (err) {
      setMessage('Error: ' + err.message);
    } finally {
      setIsSegmenting(false);
    }
  };

  const handleReset = () => {
    setPoints([]);
    setSegmentedImageUrl(null);
    setMaskedVideoUrl(null);
    setLandmarkedVideoUrl(null);
    setMaskedVideoFilename(null);
    setMessage('Points reset. Click on the image to select new points.');
    setCurrentStep(1);
  };

  const handleAcceptSegmentation = async () => {
    if (!points.length || !videoFilename) {
      setMessage('Error: No points or video filename available');
      return;
    }
    
    setIsProcessingVideo(true);
    setMessage('Processing full video with SAM2... This may take a few minutes.');
    
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/process_video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          points: points,
          video_filename: videoFilename,
          s3_folder: s3Folder // Pass S3 folder
        }),
      });
      
      const data = await res.json();
      
      if (res.ok) {
        setMessage('Video processing completed.');
        let url = data.masked_video_url;
        if (url && !url.startsWith('http')) {
          url = `${process.env.NEXT_PUBLIC_API_URL}${url}`;
        }
        setMaskedVideoUrl(url);
        setMaskedVideoFilename(data.masked_video_filename);
        setCurrentStep(3);
      } else {
        setMessage('Error: ' + (data.error || 'Video processing failed'));
      }
    } catch (err) {
      setMessage('Error: ' + err.message);
    } finally {
      setIsProcessingVideo(false);
    }
  };

  const handleRunOpenPose = async () => {
    if (!maskedVideoUrl || !maskedVideoFilename) {
      setMessage('Error: No masked video available');
      return;
    }
    
    setIsRunningOpenPose(true);
    setMessage('Running OpenPose on masked video... This may take a few minutes.');
    
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/run_openpose_on_masked_video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          masked_video_filename: maskedVideoFilename,
          s3_folder: s3Folder // Pass S3 folder
        }),
      });
      
      const data = await res.json();
      
      if (res.ok) {
        setMessage('OpenPose processing completed.');
        let url = data.landmarked_video_url;
        if (url && !url.startsWith('http')) {
          url = `${process.env.NEXT_PUBLIC_API_URL}${url}`;
        }
        setLandmarkedVideoUrl(url);
        setCurrentStep(4);
      } else {
        setMessage('Error: ' + (data.error || 'OpenPose processing failed'));
      }
    } catch (err) {
      setMessage('Error: ' + err.message);
    } finally {
      setIsRunningOpenPose(false);
    }
  };

  const steps = [
    { title: 'Upload Video', active: currentStep >= 0 },
    { title: 'Select Points', active: currentStep >= 1 },
    { title: 'Segment', active: currentStep >= 2 },
    { title: 'Process Video', active: currentStep >= 3 },
    { title: 'Add Landmarks', active: currentStep >= 4 }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Gait Analysis System
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload a video, select points on the patient, and generate masked videos with pose landmarks for gait analysis
          </p>
        </div>

        {/* Progress Steps */}
        <div className="flex justify-center mb-8">
          <div className="flex space-x-4">
            {steps.map((step, index) => (
              <div key={index} className="flex items-center">
                <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 ${
                  step.active 
                    ? 'bg-blue-600 text-white border-transparent' 
                    : 'bg-white text-gray-400 border-gray-300'
                }`}>
                  <span className="text-sm font-medium">{index + 1}</span>
                </div>
                {index < steps.length - 1 && (
                  <div className={`w-12 h-0.5 mx-2 ${step.active ? 'bg-blue-600' : 'bg-gray-300'}`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
            {/* Upload Section */}
            <div className="mb-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">Upload Video</h2>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <div className="text-4xl mb-4 text-gray-400">📹</div>
                    <p className="text-lg font-medium text-gray-700 mb-2">
                      {file ? file.name : 'Choose a video file'}
                    </p>
                    <p className="text-gray-500">MP4, MOV, AVI, or other video formats</p>
                  </label>
                </div>
                <button
                  type="submit"
                  disabled={!file || message === 'Uploading...'}
                  className="w-full bg-blue-600 text-white font-semibold text-lg rounded-lg py-3 shadow-sm hover:bg-blue-700 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {message === 'Uploading...' ? 'Uploading...' : 'Upload Video'}
                </button>
              </form>
            </div>

            {/* Status Message */}
            {message && (
              <div className={`mb-6 p-4 rounded-lg ${
                message.includes('Error') ? 'bg-red-50 text-red-800 border border-red-200' :
                message.includes('completed') ? 'bg-green-50 text-green-800 border border-green-200' :
                'bg-blue-50 text-blue-800 border border-blue-200'
              }`}>
                <p className="font-medium">{message}</p>
              </div>
            )}

            {/* First Frame Section */}
            {firstFrameUrl && (
              <div className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Select Points</h2>
                <div className="bg-gray-50 rounded-lg p-6">
                  <p className="text-gray-600 mb-4">Click on a few points on the patient for segmentation:</p>
                  <div className="relative inline-block">
                    <img
                      ref={setImageRef}
                      src={firstFrameUrl}
                      alt="First frame"
                      className="max-w-full h-auto rounded-lg cursor-crosshair border border-gray-300 hover:border-blue-400 transition-colors"
                      onClick={handleImageClick}
                    />
                    <svg
                      className="absolute top-0 left-0 pointer-events-none"
                      width="100%"
                      height="100%"
                      preserveAspectRatio="none"
                    >
                      {points.map((pt, idx) => {
                        if (!imageRef) return null;
                        const displayX = (pt.x / imageRef.naturalWidth) * imageRef.offsetWidth;
                        const displayY = (pt.y / imageRef.naturalHeight) * imageRef.offsetHeight;
                        return (
                          <circle key={idx} cx={displayX} cy={displayY} r="6" fill="#dc2626" stroke="white" strokeWidth="2" />
                        );
                      })}
                    </svg>
                  </div>
                  
                  {points.length > 0 && (
                    <div className="mt-6 space-y-4">
                      <div className="bg-white rounded-lg p-4 border border-gray-200">
                        <h3 className="font-semibold text-gray-900 mb-2">Selected Points ({points.length}):</h3>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                          {points.map((pt, idx) => (
                            <div key={idx} className="text-sm bg-gray-100 rounded px-2 py-1">
                              Point {idx + 1}: ({pt.x}, {pt.y})
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex gap-3">
                        <button
                          onClick={handleSegment}
                          disabled={isSegmenting}
                          className="flex-1 bg-green-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
                        >
                          {isSegmenting ? 'Segmenting...' : 'Run SAM2 Segmentation'}
                        </button>
                        <button
                          onClick={handleReset}
                          className="bg-gray-500 text-white font-semibold py-3 px-6 rounded-lg hover:bg-gray-600 transition-colors"
                        >
                          Reset
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Segmentation Result */}
            {segmentedImageUrl && (
              <div className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Segmentation Result</h2>
                <div className="bg-gray-50 rounded-lg p-6">
                  <img
                    src={segmentedImageUrl}
                    alt="Segmented frame"
                    className="max-w-full h-auto rounded-lg border border-gray-200"
                  />
                  <div className="mt-6 flex gap-3">
                    <button
                      onClick={handleAcceptSegmentation}
                      disabled={isProcessingVideo}
                      className="flex-1 bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
                    >
                      {isProcessingVideo ? 'Processing...' : 'Process Full Video'}
                    </button>
                    <button
                      onClick={handleReset}
                      className="bg-red-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-red-700 transition-colors"
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Masked Video Result */}
            {maskedVideoUrl && (
              <div className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Masked Video</h2>
                <div className="bg-gray-50 rounded-lg p-6">
                  <video
                    controls
                    className="w-full rounded-lg border border-gray-200"
                  >
                    <source src={maskedVideoUrl} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                  <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
                    <p className="text-green-800 font-medium">Video processing completed</p>
                    <p className="text-green-700 text-sm mt-1">The masked video shows only the selected anatomical region.</p>
                  </div>
                  <div className="mt-6">
                    <button
                      onClick={handleRunOpenPose}
                      disabled={isRunningOpenPose || !maskedVideoFilename}
                      className="w-full bg-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50"
                    >
                      {isRunningOpenPose ? 'Running OpenPose...' : 'Add Pose Landmarks'}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Final Video with Landmarks */}
            {landmarkedVideoUrl && (
              <div className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Video with Pose Landmarks</h2>
                <div className="bg-gray-50 rounded-lg p-6">
                  <video
                    controls
                    className="w-full rounded-lg border border-gray-200"
                  >
                    <source src={landmarkedVideoUrl} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                  <div className="mt-4 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                    <p className="text-indigo-800 font-medium">OpenPose processing completed</p>
                    <p className="text-indigo-700 text-sm mt-1">The video now shows pose landmarks overlaid on the masked region for gait analysis.</p>
                  </div>
                  <div className="mt-4 text-center">
                    <p className="text-gray-600 text-sm">Right-click the video and select "Save video as..." to download</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}