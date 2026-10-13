const form = document.getElementById('house-form');
const button = document.getElementById('estimate-button');
const errorBox = document.getElementById('error');
const currency = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
let fields = [];

async function requestApi(path, options) {
  if (window.location.protocol === 'file:') {
    throw new Error('This HTML file cannot run predictions by itself. Start the backend with .\\.venv\\Scripts\\python.exe app.py, then open http://127.0.0.1:8000.');
  }
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error('Cannot reach the prediction backend. Run .\\.venv\\Scripts\\python.exe app.py, keep that terminal open, and visit http://127.0.0.1:8000 instead of Live Server or the HTML file.');
  }
  if (!(response.headers.get('content-type') || '').includes('application/json')) {
    throw new Error('This page is not connected to the Python backend. Start app.py and open http://127.0.0.1:8000, not a separate preview server.');
  }
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'The server could not complete the request.');
  return result;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearResult() {
  document.getElementById('price').textContent = '—';
  document.getElementById('result-label').textContent = 'Ready for a fresh estimate';
  document.getElementById('result-note').textContent = 'Submit your updated details to see a new estimate.';
}

async function initialize() {
  try {
    fields = (await requestApi('/api/fields')).fields;
    for (const field of fields) {
      const label = document.createElement('label');
      label.textContent = field.label;
      const input = document.createElement(field.type === 'number' ? 'input' : 'select');
      input.name = field.name;
      if (field.type === 'number') {
        input.type = 'number';
        input.min = field.min;
        input.max = field.max;
        input.step = 'any';
        input.value = field.default;
        input.defaultValue = field.default;
        input.required = true;
      } else {
        for (const value of field.options) {
          const option = new Option(value, value, value === field.default, value === field.default);
          input.add(option);
        }
        input.add(new Option('Not specified / missing', ''));
      }
      label.appendChild(input);
      if (field.type === 'number') {
        const range = document.createElement('span');
        range.className = 'range';
        range.textContent = `Training range: ${field.min.toLocaleString()}–${field.max.toLocaleString()}`;
        label.appendChild(range);
      }
      document.getElementById(field.advanced ? 'advanced-fields' : 'core-fields').appendChild(label);
    }
    button.disabled = false;
  } catch (error) {
    showError(error.message);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  clearResult();
  button.disabled = true;
  button.textContent = 'Finding your estimate…';
  try {
    const values = {};
    for (const field of fields) {
      const value = form.elements.namedItem(field.name).value;
      values[field.name] = value === '' ? null : field.type === 'number' ? Number(value) : value;
    }
    const result = await requestApi('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values),
    });
    document.getElementById('price').textContent = currency.format(result.price);
    document.getElementById('result-label').textContent = 'Your estimated sale price';
    document.getElementById('result-note').textContent = 'Based on your details and historical Ames housing patterns. No confidence interval is available.';
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = 'Estimate my home’s price ↗';
  }
});

form.addEventListener('reset', () => { errorBox.hidden = true; clearResult(); });
form.addEventListener('input', clearResult);
initialize();
