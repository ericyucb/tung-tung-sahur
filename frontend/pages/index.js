import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setMessage('Uploading...');
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setMessage('Upload successful: ' + data.filename);
      } else {
        setMessage('Error: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      setMessage('Error: ' + err.message);
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
      </div>
    </div>
  );
} 