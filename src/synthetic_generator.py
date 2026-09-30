import os
import numpy as np
import pandas as pd

def generate_mock_demographic_data(output_path: str = "data/synthetic_population.parquet", num_countries: int = 15, start_year: int = 1990, end_year: int = 2025):
    """
    Generates synthetic demographic tensors matching UN WPP longitudinal distributions.
    Saves outputs as parquet format for offline CI caching.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    np.random.seed(42)
    countries = [f"Region_Country_{i:02d}" for i in range(1, num_countries + 1)]
    years = list(range(start_year, end_year + 1))
    
    records = []
    for c in countries:
        base_pop = np.random.uniform(5e6, 60e6)
        growth_rate = np.random.uniform(0.015, 0.032)
        fertility_rate = np.random.uniform(3.5, 6.5)
        mortality_rate = np.random.uniform(0.006, 0.012)
        
        current_pop = base_pop
        for y in years:
            noise = np.random.normal(0, 0.002)
            effective_growth = growth_rate - (y - start_year) * 0.0003 + noise
            current_pop = current_pop * (1 + effective_growth)
            records.append({
                "country": c,
                "year": y,
                "population": int(current_pop),
                "fertility_rate": round(max(1.5, fertility_rate - (y - start_year) * 0.05 + np.random.normal(0, 0.05)), 2),
                "crude_death_rate": round(max(0.003, mortality_rate - (y - start_year) * 0.0001 + np.random.normal(0, 0.0005)), 4),
                "urbanization_pct": round(min(80.0, 20.0 + (y - start_year) * 0.7 + np.random.normal(0, 0.2)), 2)
            })
            
    df = pd.DataFrame(records)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Generated {len(df)} synthetic demographic records saved to: {output_path}")
    return df

if __name__ == "__main__":
    generate_mock_demographic_data()
