(function () {
  // Toggle show/hide for any eye button sitting next to an input
  document.querySelectorAll('.toggle-eye').forEach(btn => {
    const input = btn.previousElementSibling;
    if (!input) return;
    btn.addEventListener('click', () => {
      const isPwd = input.type === 'password';
      input.type = isPwd ? 'text' : 'password';
      btn.textContent = isPwd ? '🙈' : '👁️';
    });
  });

  // Optional: wire placeholders if IDs exist (login form usually has these)
  const u = document.getElementById('id_username');
  const p = document.getElementById('id_password') || document.getElementById('id_password1');
  if (u){ u.placeholder = '👤 Your username'; u.autocomplete = 'username'; }
  if (p){ p.placeholder = '🔒 Your password'; p.autocomplete = p.id === 'id_password' ? 'current-password' : 'new-password'; }

  // Optional: CapsLock warning if element exists
  const warn = document.getElementById('capsWarn');
  if (p && warn){
    p.addEventListener('keyup', e => { warn.style.display = e.getModifierState && e.getModifierState('CapsLock') ? 'block' : 'none'; });
    p.addEventListener('blur', () => warn.style.display = 'none');
  }
})();
