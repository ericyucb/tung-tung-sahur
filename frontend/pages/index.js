import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [firstFrameUrl, setFirstFrameUrl] = useState(null);
  const [points, setPoints] = useState([]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setFirstFrameUrl(null);
    setPoints([]);
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
    const x = Math.round(e.nativeEvent.clientX - rect.left);
    const y = Math.round(e.nativeEvent.clientY - rect.top);
    const newPoints = [...points, { x, y }];
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
                viewBox={`0 0 400 225`}
                preserveAspectRatio="none"
              >
                {points.map((pt, idx) => (
                  <circle key={idx} cx={pt.x} cy={pt.y} r={5} fill="red" />
                ))}
              </svg>
            </div>
            <div className="mt-4 text-left">
              <h3 className="font-semibold">Selected Points:</h3>
              <ul className="text-sm">
                {points.map((pt, idx) => (
                  <li key={idx}>({pt.x}, {pt.y})</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 