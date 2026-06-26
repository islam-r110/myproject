import sys

try:
    import main
    import pandas as pd
    import matplotlib
    import seaborn
    import streamlit
except Exception as e:
    print('IMPORT_ERROR', repr(e))
    sys.exit(1)

try:
    df = main.load_data('dubai_real_estate_data_realistic_500.csv')
except Exception as e:
    print('LOAD_ERROR', repr(e))
    sys.exit(1)

print('DF_TYPE', type(df).__name__)
print('HAS_AREA', 'area' in df.columns)
print('HAS_LEAD_SOURCE', 'lead_source' in df.columns)
print('HAS_SALE_AMOUNT', 'sale_amount_aed' in df.columns)
print('HAS_PROPERTY_TYPE', 'property_type' in df.columns)

for fn_name in [
    'plot_financial_volume_by_location',
    'plot_marketing_channels_roi',
    'plot_monthly_sales_performance',
]:
    try:
        fig = getattr(main, fn_name)(df)
        print(fn_name, isinstance(fig, matplotlib.figure.Figure))
    except Exception as e:
        print(fn_name + '_ERROR', repr(e))
        sys.exit(1)

if 'area' in df.columns and not df['area'].dropna().empty:
    area = df['area'].dropna().iloc[0]
    try:
        fig = main.plot_property_type_composition(df, area)
        print('plot_property_type_composition', isinstance(fig, matplotlib.figure.Figure))
    except Exception as e:
        print('plot_property_type_composition_ERROR', repr(e))
        sys.exit(1)
else:
    print('plot_property_type_composition', 'SKIPPED_NO_AREA')

print('VALIDATION_COMPLETE')
