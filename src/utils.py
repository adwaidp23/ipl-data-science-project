TEAM_STANDARDIZATION = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Kings XI Punjab': 'Punjab Kings'
}

def standardize_teams(df, columns):
    """Apply team name standardization"""
    for col in columns:
        df[col] = df[col].replace(TEAM_STANDARDIZATION)
    return df