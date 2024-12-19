from fastapi import FastAPI, Query, HTTPException
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
import uvicorn

TIME = 21

# Enums for categorical fields
class Priority(str, Enum):
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    EMPTY = ""


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"
    ALPHABETICAL = "alphabetical"


# Request Models
class PredictionRequest(BaseModel):
    sku_number: str
    brand: str
    product_category: str
    product_subcategory: str
    seller_name: str
    priority: Priority
    total_units_prev: float = Field(gt=0)
    lms_company_type: str
    time: int = Field(ge=-93, le=90)

    class Config:
        json_schema_extra = {
            "example": {
                "sku_number": "XPH00359",
                "brand": "garnier",
                "product_category": "skin_care",
                "product_subcategory": "masks_&_exfoliators",
                "seller_name": "l'oreal_philippines",
                "priority": "p1",
                "total_units_prev": 1000.0,
                "lms_company_type": "principal",
                "time": 0,
            }
        }


class GlobalRankingsRequest(BaseModel):
    data: List[PredictionRequest]


# Response Models
class BrandScore(BaseModel):
    brand: str
    priority: str
    depletion_percent: float


class GlobalRankingsResponse(BaseModel):
    data: List[BrandScore]
    total: int
    limit: int


class SellerBrandScore(BaseModel):
    seller_name: str
    brand: str
    depletion_percent: float


class SellerBrandsResponse(BaseModel):
    seller: str
    data: List[SellerBrandScore]
    total: int


class SellerSkuScore(BaseModel):
    seller_name: str
    sku_number: str
    depletion_percent: float


class SellerSkusResponse(BaseModel):
    seller: str
    data: List[SellerSkuScore]
    total: int

class VariantData(BaseModel):
    variant_id: str
    variant_sku: str
    sku: str
    name: str
    brand: str
    product_category: str
    product_sub_category: str
    priority: str
    total_units_prev: int
    persona_seller_type: str

class ScorePredictionRequest(BaseModel):
    lms_company_id: str
    lms_company_name: str
    variants: List[VariantData]

class ModelResponseData(VariantData):
    prediction_score: float


class VariantDataRequest(BaseModel):
    variant_id: str
    variant_sku: str
    sku: str
    name: str
    brand: str
    product_category: str
    product_sub_category: str
    # priority: str
    # total_units_prev: int


class VariantDataResponse(VariantDataRequest):
    score: float


class ScorePredictionRequest(BaseModel):
    lms_company_id: str
    lms_company_name: str
    lms_company_type: str
    variants: List[VariantDataRequest]


class ScorePredictionResponse(BaseModel):
    lms_company_id: str
    lms_company_name: str
    lms_company_type: str
    variants: List[VariantDataResponse]


def post_processing(predictions):
    for i in range(len(predictions)):
        if predictions[i] >= 1:
            predictions[i] = float(np.random.uniform(low=90, high=100))
        if predictions[i] <= 0:
            predictions[i] = float(np.random.uniform(low=0, high=10))
    return predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model and data
    app.state.model = joblib.load("depletion_model/model.pkl")
    app.state.train_cols = joblib.load("depletion_model/train_cols.pkl")
    app.state.X_val = joblib.load("depletion_model/X_val.pkl")

    # Pre-compute global rankings at startup
    df = pd.DataFrame(app.state.X_val, columns=app.state.train_cols)
    predictions = app.state.model.predict(app.state.X_val)
    predictions = post_processing(predictions)
    df["scores"] = predictions
    app.state.global_rankings_df = df  # Store for global use

    yield


app = FastAPI(
    title="Brand Rankings API",
    description="API for predicting and ranking brand and SKU depletion scores",
    version="1.0.0",
    lifespan=lifespan,
)


def get_seller_data(seller_name: str, df: pd.DataFrame):
    seller_data = df[df["seller_name"] == seller_name]
    if seller_data.empty:
        raise HTTPException(status_code=404, detail="Seller not found")
    return seller_data


@app.get(
    "/pollen/depletion_model/brand_index/global",
    response_model=GlobalRankingsResponse,
)
async def get_global_brand_scores(
    limit: Optional[int] = Query(10, gt=0, le=100),
    sort_order: SortOrder = Query(SortOrder.DESC),
):
    global_brands = joblib.load("depletion_model/global_brand_ranking.pkl")

    return {
        "data": global_brands.head(limit).to_dict("records"),
        "total": len(global_brands),
        "limit": limit,
    }


@app.get(
    "/pollen/depletion_model/brand_index/by_seller/{seller_name}",
    # response_model=SellerBrandsResponse,
)
async def get_seller_brand_scores(seller_name: str):
    seller_brand_wise_ranking = joblib.load(
        "depletion_model/seller_brand_wise_ranking.pkl"
    )
    seller_brands = seller_brand_wise_ranking[
        seller_brand_wise_ranking["seller_name"] == seller_name
    ]

    return {
        "seller": seller_name,
        "data": seller_brand_wise_ranking.to_dict("records"),
        "total": len(seller_brands),
    }


@app.get(
    "/pollen/depletion_model/depletion_score/by_seller/{seller_name}",
    # response_model=SellerSkusResponse,
)
async def get_seller_product_scores(seller_name: str):
    seller_sku_wise_ranking = joblib.load(
        "depletion_model/seller_sku_wise_ranking.pkl"
    )
    seller_skus = seller_sku_wise_ranking[
        seller_sku_wise_ranking["seller_name"] == seller_name
    ]

    return {
        "seller": seller_name,
        "data": seller_skus.to_dict("records"),
        "total": len(seller_skus),
    }


@app.post(
    "/pollen/depletion_model/depletion_score/by_seller/{seller_name}",
    response_model=ScorePredictionResponse,
)
async def model_inference(inference_request_data: ScorePredictionRequest):
    model = joblib.load("depletion_model/model.pkl")
    time = TIME

    # Create a list to store scored variants
    scored_variants = []

    # Process each variant
    for variant_data in inference_request_data.variants:
        # Create prediction request data
        request = [
            variant_data.sku,
            variant_data.brand,
            variant_data.product_category,
            variant_data.product_sub_category,
            inference_request_data.lms_company_name,
            "P1",
            100,
            inference_request_data.lms_company_type,
            time,
        ]

        # Make prediction
        prediction_score = float(model.predict([request])[0])

        # Post process prediction if needed
        prediction_score = post_processing([prediction_score])[0]

        # Create response variant with score
        scored_variant = VariantDataResponse(
            **variant_data.model_dump(), score=prediction_score
        )

        scored_variants.append(scored_variant)

    # Create response
    response = ScorePredictionResponse(
        lms_company_id=inference_request_data.lms_company_id,
        lms_company_name=inference_request_data.lms_company_name,
        lms_company_type=inference_request_data.lms_company_type,
        variants=scored_variants,
    )

    return response


if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)
