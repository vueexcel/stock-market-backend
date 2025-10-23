from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from bigquery_data import generate_big_query_data
from dotenv import load_dotenv
from io import BytesIO
from datetime import datetime, date
from uuid import uuid4
from typing import Dict, Any, List
import os
from analytics_data import calculate_all_returns
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

@app.route('/analytics_performance', methods=['POST', 'OPTIONS', 'GET'])
def analytics_performance():
    try:
        if request.method == 'OPTIONS':
            return ('', 204)
        data = request.get_json(force=True, silent=True) or {}
        ticker = (data.get('ticker') or '').strip()
        if not ticker:
            return jsonify({'success': False, 'error': 'Missing required field: ticker'}), 400
        # Optional custom date range (expects ISO dates from HTML date inputs)
        custom_start_raw = (data.get('customStartDate') or '').strip()
        custom_end_raw = (data.get('customEndDate') or '').strip()
        custom_range = None
        if custom_start_raw and custom_end_raw:
            try:
                start_dt = date.fromisoformat(custom_start_raw)
                end_dt = date.fromisoformat(custom_end_raw)
                if start_dt <= end_dt:
                    custom_range = (start_dt, end_dt)
            except Exception:
                # Ignore invalid date formats and proceed without custom range
                custom_range = None

        results = calculate_all_returns(ticker=ticker, custom_range=custom_range)
        def format_periods(periods, label_key='period', label_prefix=None):
            formatted = []
            for row in periods:
                period = row.get(label_key)
                if label_prefix:
                    period = f"{label_prefix}: {period}"
                start_date = row.get('start_date_found') or row.get('start_date_requested')
                end_date = row.get('end_date_found') or row.get('end_date_requested')
                years = float(row.get('years') or 0.0)
                start_price = float(row.get('start_price') or 0.0)
                end_price = float(row.get('end_price') or 0.0)
                price_diff = float(row.get('price_difference') or (end_price - start_price))
                total_return = float(row.get('total_return_pct') or (
                    0.0 if start_price == 0 else round(((end_price - start_price) / start_price) * 100, 2)
                ))
                simple_annual_return = round(total_return / years, 2) if years > 0 else 0.0
                cagr_percent = round((((end_price / start_price) ** (1 / years)) - 1) * 100, 2) if (years > 0 and start_price > 0) else 0.0
                formatted.append({
                    'period': period,
                    'startDate': start_date,
                    'endDate': end_date,
                    'years': years,
                    'startPrice': start_price,
                    'endPrice': end_price,
                    'priceDifference': price_diff,
                    'totalReturn': total_return,
                    'simpleAnnualReturn': simple_annual_return,
                    'cagrPercent': cagr_percent
                })
            return formatted
        dynamic_data = format_periods(results.get('dynamic_periods', []))
        predefined_data = format_periods(results.get('predefined_periods', []))
        annual_data = format_periods(results.get('annual_returns', []), label_key='year')
        # Custom range (single row)
        custom_raw = results.get('custom_range')
        custom_data = []
        if custom_raw:
            # Normalize shape to match other rows and add a fixed label
            start_date = custom_raw.get('start_date_found') or custom_raw.get('start_date_requested')
            end_date = custom_raw.get('end_date_found') or custom_raw.get('end_date_requested')
            years = float(custom_raw.get('years') or 0.0)
            start_price = float(custom_raw.get('start_price') or 0.0)
            end_price = float(custom_raw.get('end_price') or 0.0)
            price_diff = float(custom_raw.get('price_difference') or (end_price - start_price))
            total_return = float(custom_raw.get('total_return_pct') or (
                0.0 if start_price == 0 else round(((end_price - start_price) / start_price) * 100, 2)
            ))
            simple_annual_return = round(total_return / years, 2) if years > 0 else 0.0
            cagr_percent = round((((end_price / start_price) ** (1 / years)) - 1) * 100, 2) if (years > 0 and start_price > 0) else 0.0
            custom_data.append({
                'period': 'Selected dates',
                'startDate': start_date,
                'endDate': end_date,
                'years': years,
                'startPrice': start_price,
                'endPrice': end_price,
                'priceDifference': price_diff,
                'totalReturn': total_return,
                'simpleAnnualReturn': simple_annual_return,
                'cagrPercent': cagr_percent
            })

        # Structured response
        response = {
            'success': True,
            'ticker': results.get('ticker', ticker.upper()),
            'asOfDate': results.get('as_of_date'),
            'performance': {
                'dynamicPeriods': dynamic_data,
                'predefinedPeriods': predefined_data,
                'annualReturns': annual_data,
                'customRange': custom_data
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error processing analytics_performance: {str(e)}")
        return jsonify({'success': False, 'error': f'Failed to get analytics performance: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
