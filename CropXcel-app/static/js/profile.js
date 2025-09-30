// CSRF cookie helper
function getCookie(name){
  const m = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
  return m ? m.pop() : '';
}
const form = document.getElementById('avatarForm');
const file = document.getElementById('avatarFile');
const img  = document.getElementById('avatarImg');
const fallback = document.getElementById('avatarFallback');
const prog = document.getElementById('prog');
const bar  = document.getElementById('bar');
const stat = document.getElementById('stat');

// Preview before upload
file.addEventListener('change', () => {
  const f = file.files && file.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  if (fallback) fallback.style.display = 'none';
  const imgWrap = img.closest('.avatar-img');
  if (imgWrap) imgWrap.style.display = 'grid';
  img.src = url;

  // auto-submit after pick
  form.dispatchEvent(new Event('submit', { cancelable: true }));
});

// AJAX upload with progress
form.addEventListener('submit', (e) => {
  const isRemove = e.submitter && e.submitter.name === 'remove_avatar';
  if (isRemove) return; // allow normal POST if you later add remove button

  e.preventDefault();
  const data = new FormData(form);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', form.action, true);
  xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
  xhr.setRequestHeader('X-CSRFToken', getCookie('csrftoken'));

  prog.style.display = 'block';
  stat.style.display = 'block';
  bar.style.width = '0%';
  stat.textContent = 'Uploading…';

  if (xhr.upload) {
    xhr.upload.addEventListener('progress', (ev) => {
      if (ev.lengthComputable) {
        const pct = Math.round((ev.loaded / ev.total) * 100);
        bar.style.width = pct + '%';
        stat.textContent = 'Uploading… ' + pct + '%';
      }
    });
  }

  xhr.onload = () => {
    try {
      const res = JSON.parse(xhr.responseText || '{}');
      if (res.ok && res.avatar_url) {
        img.src = res.avatar_url; // server should return a cache-busted URL
        stat.textContent = 'Done!';
      } else {
        stat.textContent = res.error || 'Upload failed.';
      }
    } catch {
      stat.textContent = 'Upload failed.';
    }
    setTimeout(() => { prog.style.display = 'none'; stat.style.display = 'none'; }, 900);
  };

  xhr.onerror = () => {
    stat.textContent = 'Network error.';
    setTimeout(() => { prog.style.display = 'none'; stat.style.display = 'none'; }, 1200);
  };

  xhr.send(data);
});