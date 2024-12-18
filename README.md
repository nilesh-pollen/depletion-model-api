# Brand Rankings API

A FastAPI service for predicting and ranking brand and SKU depletion scores.

## Installation & Setup

1. Install `uv` (fast Python package installer):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv install
```

## Running the Server

Start the FastAPI server:
```bash
uv run inference.py
```

The server will be available at `http://localhost:8000`

## API Documentation

Interactive API documentation is available at: `http://localhost:8000/docs`

### Available Endpoints

1. **Global Brand Scores**
   - Endpoint: `/pollen/depletion_model/brand_index/global`
   - Method: GET
   - Parameters: 
     - `limit` (optional): Maximum number of results (default: 10, max: 100)
     - `sort_order` (optional): Sort order ("asc", "desc", "alphabetical")

2. **Seller Brand Scores**
   - Endpoint: `/pollen/depletion_model/brand_index/by_seller/{seller_name}`
   - Method: GET
   - Parameters:
     - `seller_name` (required): Name of the seller

3. **Seller Product Scores**
   - Endpoint: `/pollen/depletion_model/depletion_score/by_seller/{seller_name}`
   - Method: GET
   - Parameters:
     - `seller_name` (required): Name of the seller

### Response Models

The API returns structured JSON responses with the following models:

- `GlobalRankingsResponse`: List of brand scores globally
- `SellerBrandsResponse`: Brand scores for a specific seller
- `SellerSkusResponse`: SKU scores for a specific seller

## Project Structure
```
.
├── README.md
├── inference.py              # FastAPI server
├── pyproject.toml           # Project dependencies
├── uv.lock                  # Lock file for dependencies
└── depletion_model/        # Model artifacts
    ├── model.pkl
    ├── global_brand_ranking.pkl
    ├── seller_brand_wise_ranking.pkl
    ├── seller_sku_wise_ranking.pkl
    ├── train_cols.pkl
    ├── X_train.pkl
    ├── X_val.pkl
    ├── y_train.pkl
    └── y_val.pkl
```