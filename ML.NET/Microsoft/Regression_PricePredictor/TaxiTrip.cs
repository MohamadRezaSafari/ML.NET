﻿using Microsoft.ML.Data;

namespace Regression_PricePredictor;

public class TaxiTrip
{
    [LoadColumn(0)]
    public string? VendorId { get; set; }

    [LoadColumn(1)]
    public string? RateCode;

    [LoadColumn(2)]
    public float PassengerCount;

    [LoadColumn(3)]
    public float TripTime;

    [LoadColumn(4)]
    public float TripDistance;

    [LoadColumn(5)]
    public string? PaymentType;

    [LoadColumn(6)]
    public float FareAmount;
}

public class TaxiTripFarePrediction
{
    [ColumnName("Score")]
    public float FareAmount;
}