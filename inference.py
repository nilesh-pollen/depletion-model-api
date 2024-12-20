from fastapi import FastAPI, Query, HTTPException
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import uvicorn


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


class VariantDataResponse(VariantDataRequest):
    score: dict


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


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Load the ML model and data
#     app.state.model = joblib.load("depletion_model/model.pkl")
#     app.state.train_cols = joblib.load("depletion_model/train_cols.pkl")
#     app.state.X_val = joblib.load("depletion_model/X_val.pkl")

#     # Pre-compute global rankings at startup
#     df = pd.DataFrame(app.state.X_val, columns=app.state.train_cols)
#     predictions = app.state.model.predict(app.state.X_val)
#     predictions = post_processing(predictions)
#     df["scores"] = predictions
#     app.state.global_rankings_df = df  # Store for global use

#     yield


app = FastAPI(
    title="Brand Rankings API",
    description="API for predicting and ranking brand and SKU depletion scores",
    version="1.0.0",
    # lifespan=lifespan,
)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allows all origins - you can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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
    "/api/v1/depletion_model/product_variant_score",
    response_model=ScorePredictionResponse,
)
async def model_inference(inference_request_data: ScorePredictionRequest):
    model = joblib.load("depletion_model/model.pkl")
    low_depletion_cats = joblib.load("depletion_model/low_depletion_cats.pkl")
    mid_depletion_cats = joblib.load("depletion_model/mid_depletion_cats.pkl")

    # Create a list to store scored variants
    scored_variants = []

    # Process each variant
    for variant_data in inference_request_data.variants:
        if variant_data.product_category in low_depletion_cats:
            time = 90
        elif variant_data.product_category in mid_depletion_cats:
            time = 60
        else:
            time = 30

        # Create prediction request data
        request1 = [
            variant_data.sku,
            variant_data.brand.lower().replace(" ", "_"),
            variant_data.product_category.lower().replace(" ", "_"),
            variant_data.product_sub_category.lower().replace(" ", "_"),
            inference_request_data.lms_company_name.lower().replace(" ", "_"),
            inference_request_data.lms_company_type.lower().replace(" ", "_"),
            "p1",
            time,
        ]

        request2 = request1.copy()
        request3 = request1.copy()
        request2[6] = "p2"
        request3[6] = "p3"

        # Make prediction
        prediction_score = {
            "p1": model.predict(request1) * 100,
            "p2": model.predict(request2) * 100,
            "p3": model.predict(request3) * 100,
        }

        # Post process prediction if needed
        for priority in prediction_score.keys():
            if prediction_score[priority] >= 100:
                prediction_score[priority] = np.random.uniform(
                    low=90, high=100, size=(1,)
                )
            elif prediction_score[priority] <= 0:
                prediction_score[priority] = np.random.uniform(
                    low=0, high=10, size=(1,)
                )

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
