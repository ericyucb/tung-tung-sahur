import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [firstFrameUrl, setFirstFrameUrl] = useState(null);
  const [points, setPoints] = useState([]);
  const [segmentedImageUrl, setSegmentedImageUrl] = useState(null);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [videoFilename, setVideoFilename] = useState(null);
  const [imageRef, setImageRef] = useState(null);
  const [maskedVideoUrl, setMaskedVideoUrl] = useState(null);
  const [isRunningOpenPose, setIsRunningOpenPose] = useState(false);
  const [landmarkedVideoUrl, setLandmarkedVideoUrl] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setFirstFrameUrl(null);
    setPoints([]);
    setSegmentedImageUrl(null);
    setMaskedVideoUrl(null);
    setLandmarkedVideoUrl(null);
    setVideoFilename(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setMessage('Uploading...');
    setFirstFrameUrl(null);
    setPoints([]);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
               setMessage('Upload successful: ' + data.video_filename);
               setVideoFilename(data.video_filename);
               // Compose absolute URL for the first frame
               let url = data.first_frame_url;
               if (url && !url.startsWith('http')) {
                 // Assume backend and frontend are on same host for dev
                 url = `${process.env.NEXT_PUBLIC_API_URL}${url}`;
               }
               setFirstFrameUrl(url);
      } else {
        setMessage('Error: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      setMessage('Error: ' + err.message);
    }
  };

  const handleImageClick = async (e) => {
    if (!firstFrameUrl) return;
    const rect = e.target.getBoundingClientRect();
    const img = e.target;
    
    // Get the actual displayed dimensions
    const displayWidth = img.offsetWidth;
    const displayHeight = img.offsetHeight;
    
    // Get the natural (original) dimensions
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;
    
    // Calculate the click position relative to the displayed image
    const displayX = Math.round(e.nativeEvent.clientX - rect.left);
    const displayY = Math.round(e.nativeEvent.clientY - rect.top);
    
    // Scale the coordinates to match the original image dimensions
    const originalX = Math.round((displayX / displayWidth) * naturalWidth);
    const originalY = Math.round((displayY / displayHeight) * naturalHeight);
    
    const newPoints = [...points, { x: originalX, y: originalY }];
    setPoints(newPoints);
    
    // Send to backend
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
        setMessage('Segmentation completed!');
        // Compose absolute URL for the segmented image
        let url = data.segmented_image_url;
        if (url && !url.startsWith('http')) {
          url = `${process.env.NEXT_PUBLIC_API_URL}${url}`;
        }
        setSegmentedImageUrl(url);
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
    setMessage('Points reset. Click on the image to select new points.');
  };

  const handleAcceptSegmentation = async () => {
    if (!points.length || !videoFilename) return;
    
    setIsProcessingVideo(true);
    setMessage('Processing full video with SAM2... This may take a few minutes.');
    
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/process_video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          points: points,
          video_filename: videoFilename 
        }),
      });
      
      const data = await res.json();
      
      if (res.ok) {
        setMessage('Full video processing completed!');
        // Compose absolute URL for the masked video
        let url = data.masked_video_url;
        if (url && !url.startsWith('http')) {
          url = `${process.env.NEXT_PUBLIC_API_URL}${url}`;
        }
        setMaskedVideoUrl(url);
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
    if (!maskedVideoUrl || !videoFilename) return;
    
    setIsRunningOpenPose(true);
    setMessage('Running OpenPose on masked video... This may take a few minutes.');
    
    try {
      // Extract the masked video filename from the URL
      const maskedVideoFilename = videoFilename.replace(/\.[^/.]+$/, '_masked.mp4');
      
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/run_openpose_on_masked_video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          masked_video_filename: maskedVideoFilename
        }),
      });
      
      const data = await res.json();
      
      if (res.ok) {
        setMessage('OpenPose processing completed!');
        // Compose absolute URL for the landmarked video
        let url = data.landmarked_video_url;
        if (url && !url.startsWith('http')) {
          url = `${process.env.NEXT_PUBLIC_API_URL}${url}`;
        }
        setLandmarkedVideoUrl(url);
      } else {
        setMessage('Error: ' + (data.error || 'OpenPose processing failed'));
      }
    } catch (err) {
      setMessage('Error: ' + err.message);
    } finally {
      setIsRunningOpenPose(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-100 flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-2xl p-10 max-w-md w-full text-center">
        <h1 className="font-extrabold text-3xl mb-6 tracking-tight text-slate-800">Upload a Video</h1>
        <form onSubmit={handleSubmit} className="flex flex-col gap-5">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 border border-slate-200 rounded-lg bg-slate-50 text-base cursor-pointer"
          />
          <button
            type="submit"
            className="bg-gradient-to-r from-indigo-500 to-blue-400 text-white font-bold text-lg rounded-lg py-3 shadow-md hover:from-indigo-600 hover:to-blue-500 transition-colors duration-200"
            disabled={message === 'Uploading...'}
          >
            {message === 'Uploading...' ? 'Uploading...' : 'Upload'}
          </button>
        </form>
        {message && (
          <p className={`mt-6 font-medium ${message.startsWith('Upload successful') ? 'text-green-600' : 'text-red-600'}`}>
            {message}
          </p>
        )}
        {firstFrameUrl && (
          <div className="mt-8">
            <h2 className="font-bold mb-2">First Frame (click to select points):</h2>
            <div style={{ position: 'relative', display: 'inline-block' }}>
              <img
                ref={setImageRef}
                src={firstFrameUrl}
                alt="First frame"
                style={{ maxWidth: 400, borderRadius: 8, cursor: 'crosshair' }}
                onClick={handleImageClick}
              />
              {/* Draw points as red dots */}
              <svg
                style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
                width="100%"
                height="100%"
                preserveAspectRatio="none"
              >
                {points.map((pt, idx) => {
                  // Scale the original coordinates back to displayed size for visualization
                  if (!imageRef) return null;
                  const displayX = (pt.x / imageRef.naturalWidth) * imageRef.offsetWidth;
                  const displayY = (pt.y / imageRef.naturalHeight) * imageRef.offsetHeight;
                  return (
                    <circle key={idx} cx={displayX} cy={displayY} r={5} fill="red" />
                  );
                })}
              </svg>
            </div>
                               <div className="mt-4 text-left">
                     <h3 className="font-semibold">Selected Points:</h3>
                     <ul className="text-sm">
                       {points.map((pt, idx) => (
                         <li key={idx}>({pt.x}, {pt.y})</li>
                       ))}
                     </ul>
                     {points.length > 0 && (
                       <div className="mt-4 flex gap-2">
                         <button
                           onClick={handleSegment}
                           disabled={isSegmenting}
                           className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:opacity-50"
                         >
                           {isSegmenting ? 'Segmenting...' : 'Run SAM2 Segmentation'}
                         </button>
                         <button
                           onClick={handleReset}
                           className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600"
                         >
                           Reset Points
                         </button>
                       </div>
                     )}
                   </div>
                 </div>
               )}
               
               {segmentedImageUrl && (
                 <div className="mt-8">
                   <h2 className="font-bold mb-2">SAM2 Segmentation Result:</h2>
                   <div className="mb-4">
                     <img
                       src={segmentedImageUrl}
                       alt="Segmented frame"
                       style={{ maxWidth: 400, borderRadius: 8 }}
                     />
                   </div>
                   <div className="flex gap-2">
                     <button
                       onClick={handleAcceptSegmentation}
                       disabled={isProcessingVideo}
                       className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
                     >
                       {isProcessingVideo ? 'Processing Video...' : 'Accept Segmentation'}
                     </button>
                     <button
                       onClick={handleReset}
                       className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
                     >
                       Try Again
                     </button>
                   </div>
                 </div>
               )}
               
               {maskedVideoUrl && (
                 <div className="mt-8">
                   <h2 className="font-bold mb-2">Masked Video Result:</h2>
                   <div className="mb-4">
                     <video
                       controls
                       style={{ maxWidth: 400, borderRadius: 8 }}
                     >
                       <source src={maskedVideoUrl} type="video/mp4" />
                       Your browser does not support the video tag.
                     </video>
                   </div>
                   <div className="text-sm text-gray-600">
                     <p>✅ Video processing completed! The masked video shows only the selected region.</p>
                     <p>You can download the video by right-clicking and selecting "Save video as..."</p>
                   </div>
                   <div className="mt-4">
                     <button
                       onClick={handleRunOpenPose}
                       disabled={isRunningOpenPose}
                       className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600 disabled:opacity-50"
                     >
                       {isRunningOpenPose ? 'Running OpenPose...' : 'Add OpenPose Landmarks'}
                     </button>
                   </div>
                 </div>
        )}
        
        {landmarkedVideoUrl && (
          <div className="mt-8">
            <h2 className="font-bold mb-2">Video with OpenPose Landmarks:</h2>
            <div className="mb-4">
              <video
                controls
                style={{ maxWidth: 400, borderRadius: 8 }}
              >
                <source src={landmarkedVideoUrl} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
            <div className="text-sm text-gray-600">
              <p>✅ OpenPose processing completed! The video now shows pose landmarks.</p>
              <p>You can download the video by right-clicking and selecting "Save video as..."</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 