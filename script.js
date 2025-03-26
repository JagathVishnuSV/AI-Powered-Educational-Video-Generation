document.getElementById('videoForm').addEventListener('submit', async function(e) {
  e.preventDefault();

  const topic = document.getElementById('topic').value;
  const frames = document.getElementById('frames').value;
  const status = document.getElementById('status');
  const videoOutput = document.getElementById('videoOutput');

  status.textContent = "Generating video... Please wait.";
  videoOutput.style.display = 'none';

  try {
    const response = await fetch('http://localhost:5000/generate_video', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, frames }),
    });

    if (!response.ok) throw new Error("Failed to generate video.");
    
    const data = await response.json();
    status.textContent = "Video generated successfully!";
    videoOutput.src = data.video_url;
    videoOutput.style.display = 'block';
  } catch (error) {
    status.textContent = "Error: " + error.message;
  }
});
