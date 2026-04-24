import json

with open('d:\\workspace\\ipl-data-science-project\\static\\data.json', 'r') as f:
    data = json.load(f)

print("Keys at root:", list(data.keys()))

if 'matches' in data and len(data['matches']) > 0:
    print("\nSample match:", json.dumps(data['matches'][0], indent=2))

if 'playerStats' in data and len(data['playerStats']) > 0:
    print("\nSample playerStat:", json.dumps(data['playerStats'][0], indent=2))

if 'topTeams' in data and len(data['topTeams']) > 0:
    print("\nSample topTeam:", json.dumps(data['topTeams'][0], indent=2))

print("\nNumber of matches:", len(data.get('matches', [])))
print("Number of players:", len(data.get('playerStats', [])))
