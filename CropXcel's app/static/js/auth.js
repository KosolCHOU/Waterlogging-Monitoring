(function () {
  // Flag auth pages to allow page-specific styling (e.g. lock scrolling)
  document.body?.classList?.add('auth-page');

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

(function () {
  const wrap = document.querySelector('[data-signup-container]');
  const stepPanels = wrap ? wrap.querySelectorAll('.step-content') : null;
  if (!wrap || !stepPanels || stepPanels.length === 0) return;

  const form = wrap.querySelector('form');
  const intro = document.querySelector('.intro-screen');
  const startBtn = document.getElementById('start-signup');
  const steps = document.querySelectorAll('.progress-bar .step');
  const progressLine = document.querySelector('.progress-bar .line');
  const nextBtn = document.getElementById('next-btn');
  const backBtn = document.getElementById('back-btn');
  const step1Panel = wrap.querySelector('.step-content[data-step="1"]');
  const step2Panel = wrap.querySelector('.step-content[data-step="2"]');
  const step1Hint = document.getElementById('step1-hint');
  const step1HintDefault = step1Hint ? step1Hint.textContent.trim() : '';
  const usernameInput = document.getElementById('id_username');
  const password1Input = document.getElementById('id_password1');
  const password2Input = document.getElementById('id_password2');
  const prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const raf = window.requestAnimationFrame ? window.requestAnimationFrame.bind(window) : (cb => setTimeout(cb, 0));

  let currentStep = 1;

  const focusFirstField = (step) => {
    if (wrap.classList.contains('is-hidden')) return;
    const container = step === 2 ? step2Panel : step1Panel;
    if (!container) return;
    const field = container.querySelector('input, select, textarea');
    if (!field || typeof field.focus !== 'function') return;
    raf(() => {
      try {
        field.focus({ preventScroll: true });
      } catch (_err) {
        field.focus();
      }
    });
  };

  const scrollViewportToForm = () => {
    if (wrap.classList.contains('is-hidden')) return;
    const scrollOffset = (window.pageYOffset ?? window.scrollY ?? 0);
    const top = Math.max(0, wrap.getBoundingClientRect().top + scrollOffset - 24);
    if (prefersReducedMotion) {
      window.scrollTo(0, top);
    } else {
      window.scrollTo({ top, behavior: 'smooth' });
    }
  };

  const setActiveStep = (step) => {
    const targetStep = String(step) === '2' || Number(step) === 2 ? 2 : 1;
    currentStep = targetStep;
    stepPanels.forEach(panel => {
      const isActive = panel.dataset.step === String(targetStep);
      panel.classList.toggle('is-hidden', !isActive);
    });
    steps.forEach(node => {
      const isActive = node.dataset.step === String(targetStep);
      node.classList.toggle('active', isActive);
      if (isActive) {
        node.setAttribute('aria-current', 'step');
      } else {
        node.removeAttribute('aria-current');
      }
    });
    if (progressLine) {
      progressLine.classList.toggle('active', targetStep === 2);
    }
    clearStep1Hint();
    wrap.classList.add('ready');
    wrap.dataset.activeStep = String(targetStep);
    focusFirstField(targetStep);
    scrollViewportToForm();
  };

  const revealForm = () => {
    wrap.classList.remove('is-hidden');
    wrap.classList.add('fade-in');
    raf(() => focusFirstField(currentStep));
  };

  const clearStep1Hint = () => {
    if (!step1Hint) return;
    step1Hint.textContent = step1HintDefault || '';
    step1Hint.classList.remove('visible');
  };

  const showStep1Hint = (message) => {
    if (!step1Hint) return;
    step1Hint.textContent = (message && message.trim()) || step1HintDefault || '';
    step1Hint.classList.toggle('visible', Boolean(message));
  };

  const passwordMinLength = (() => {
    if (!password1Input) return 0;
    const attr = parseInt(password1Input.getAttribute('minlength') || '', 10);
    if (!Number.isNaN(attr) && attr > 0) return attr;
    if (typeof password1Input.minLength === 'number' && password1Input.minLength > 0) {
      return password1Input.minLength;
    }
    return 8;
  })();

  const evaluateStep1 = () => {
    const usernameValue = (usernameInput?.value || '').trim();
    const pwd1 = password1Input?.value || '';
    const pwd2 = password2Input?.value || '';
    if (!usernameValue) {
      return { ok: false, reason: 'Please choose a username before continuing.' };
    }
    if (!pwd1) {
      return { ok: false, reason: 'Enter a password to continue.' };
    }
    if (passwordMinLength > 0 && pwd1.length < passwordMinLength) {
      return { ok: false, reason: `Password must be at least ${passwordMinLength} characters.` };
    }
    if (!pwd2) {
      return { ok: false, reason: 'Please confirm your password.' };
    }
    if (pwd1 !== pwd2) {
      return { ok: false, reason: 'Passwords must match before continuing.' };
    }
    return { ok: true, reason: '' };
  };

  const reportStep1Fields = () => {
    usernameInput?.reportValidity?.();
    password1Input?.reportValidity?.();
    password2Input?.reportValidity?.();
  };

  const step2HasErrors = !!step2Panel?.querySelector('.field-error');
  const step1HasErrors = !!step1Panel?.querySelector('.field-error');
  const initialStep = step2HasErrors ? 2 : 1;

  setActiveStep(initialStep);

  const showFormImmediately = !intro || !startBtn || step1HasErrors || step2HasErrors;
  if (showFormImmediately) {
    revealForm();
  }

  if (intro && startBtn && !showFormImmediately) {
    startBtn.addEventListener('click', () => {
      intro.classList.add('is-hidden');
      setTimeout(() => intro.remove(), 400);
      revealForm();
      setActiveStep(1);
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      const result = evaluateStep1();
      if (result.ok) {
        clearStep1Hint();
        setActiveStep(2);
      } else if (step1Hint) {
        showStep1Hint(result.reason);
        reportStep1Fields();
      }
    });
  }

  if (backBtn) {
    backBtn.addEventListener('click', () => setActiveStep(1));
  }

  [usernameInput, password1Input, password2Input].forEach(input => {
    if (!input) return;
    input.addEventListener('input', () => {
      if (!step1Hint || !step1Hint.classList.contains('visible')) return;
      const result = evaluateStep1();
      if (result.ok) {
        clearStep1Hint();
      } else {
        showStep1Hint(result.reason);
      }
    });
  });

  if (form) {
    form.addEventListener('submit', event => {
      if (currentStep === 1) {
        event.preventDefault();
        const result = evaluateStep1();
        if (result.ok) {
          clearStep1Hint();
          setActiveStep(2);
        } else if (step1Hint) {
          showStep1Hint(result.reason);
          reportStep1Fields();
        }
      }
    });
  }
})();
