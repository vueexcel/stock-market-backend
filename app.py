from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from bigquery_data import generate_big_query_data
from dotenv import load_dotenv
from io import BytesIO
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any, List
import os

load_dotenv()

app = Flask(__name__)
HOSTS = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173,*').split(',')
SECRET_KEY = os.getenv('FLASK_SECRET_KEY', uuid4().hex)

app.config['SECRET_KEY'] = SECRET_KEY
CORS(app, resources={r"/*": {"origins": HOSTS}}, supports_credentials=True)


def _make_filename(prefix: str, ext: str = 'xlsx') -> str:
    """Creates a unique filename with a timestamp."""
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid4().hex[:6]
    return f"{prefix}_{now}_{unique_id}.{ext}"

@app.route('/download_bigquery_data', methods=['POST'])
def download_bigquery_data():
    """
    Endpoint to trigger fetching data from BigQuery, uploading to Google Drive,
    and returning the file for download.
    """
    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    
    tickers: List[str] = data.get('bigquery_ticker', [])
    if not isinstance(tickers, list) or not tickers:
        return jsonify({'success': False, 'error': 'Please provide at least one ticker as a list.'}), 400

    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    if not tickers:
        return jsonify({'success': False, 'error': 'No valid tickers provided.'}), 400

    start_date = (data.get('bigquery_start_date') or '').strip()
    end_date = (data.get('bigquery_end_date') or '').strip()
    if not start_date or not end_date:
        return jsonify({'success': False, 'error': 'Please provide both start and end dates.'}), 400

    interval = (data.get('bigquery_interval') or '1d').strip()

    try:
        tickers_dict = {'Custom': tickers}
        # This function uploads to Drive and returns the file in memory
        output_buffer: BytesIO = generate_big_query_data(
            start_date=start_date,
            end_date=end_date,
            tickers=tickers_dict,
            multisheet=False, # Hardcoded to match original app behavior
            as_csv=False,     # Hardcoded to match original app behavior
            interval=interval
        )

        if not output_buffer:
            return jsonify({'success': False, 'error': 'No data found for the given parameters.'}), 404

        filename = _make_filename('BigQueryData', 'xlsx')
        
        output_buffer.seek(0)
        return send_file(
            output_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
