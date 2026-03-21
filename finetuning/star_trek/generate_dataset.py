#!/usr/bin/env python3
"""
Generate a combined guard dataset for Star Trek topic classification.
Produces up to 2,500 related and up to 2,500 unrelated questions, written to a single JSONL file.

═══════════════════════════════════════════════════════════════════════════════
AGENT CUSTOMISATION GUIDE
═══════════════════════════════════════════════════════════════════════════════
To adapt this script for a different topic (e.g. car racing, cooking, football):

STEP 1 — Update this module docstring to reflect the new topic and domain.

STEP 2 — In generate_related_questions(), replace ALL entity lists and question
         templates with ones relevant to the new topic. See the inline
         [AGENT] comments throughout that function for specific guidance at
         each section.

STEP 3 — In generate_unrelated_questions(), the existing content (cities,
         animals, tech companies, etc.) can generally be reused as-is because
         it is already off-topic for almost any domain. However, double-check
         that none of the unrelated questions accidentally overlap with the new
         topic (e.g. if adapting for "Formula 1", remove motorsport questions
         from the sports list in generate_unrelated_questions).

STEP 4 — In main(), update the output_file variable to a name that reflects
         the new topic (e.g. "car_racing_guard_dataset.jsonl").

STEP 5 — Update the filler questions inside the while-loop near the bottom of
         generate_related_questions() so they reference the new topic instead
         of "Star Trek", "stardates", and "Rules of Acquisition".

General rules:
  • Keep the {"input": <question>, "label": "related"|"not_related"} schema.
  • Aim for diversity: vary question templates so the model sees many phrasings.
  • The cap of 2,500 per class is a guideline — adjust MAX_RELATED / MAX_UNRELATED
    constants if you need more or fewer examples.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import random

# [AGENT] If you want to change the maximum number of examples per class,
# update these two constants and replace the hard-coded 2500 values below.
MAX_RELATED = 4000
MAX_UNRELATED = 4000


# ---------------------------------------------------------------------------
# Related questions
# ---------------------------------------------------------------------------

def generate_related_questions():
    questions = []

    # [AGENT] ── ENTITY GROUP 1 ─────────────────────────────────────────────
    # Purpose : The first major entity group. Here it is TOS (original series)
    #           characters. For a new topic, replace with the primary "cast" of
    #           entities — e.g. for car racing: famous drivers; for cooking:
    #           renowned chefs; for football: iconic players.
    # Pattern : Build a list of entity names, then append 2-4 templated
    #           questions per entity. Add conditional blocks for well-known
    #           entities that warrant extra specific questions.
    # Action  : Replace the list contents and the question templates below.
    # ──────────────────────────────────────────────────────────────────────
    # Characters - TOS
    tos_characters = [
        "James T. Kirk", "Spock", "Leonard McCoy", "Montgomery Scott", "Hikaru Sulu",
        "Pavel Chekov", "Nyota Uhura", "Christine Chapel", "Janice Rand", "Gary Mitchell",
        "Khan Noonien Singh", "Harry Mudd", "Trelane", "Kor", "Kang", "Koloth",
        "Garth of Izar", "Matt Decker", "Robert April", "Christopher Pike"
    ]

    for char in tos_characters:
        questions.append(f'What is the role of {char} in Star Trek?')
        questions.append(f'Who portrayed {char} in Star Trek?')
        if "Kirk" in char:
            questions.append(f'What is {char}\'s middle name?')
            questions.append(f'What is {char}\'s famous catchphrase?')
        if "Spock" in char:
            questions.append(f'What is {char}\'s Vulcan heritage?')
            questions.append(f'What is {char}\'s famous hand gesture?')

    # [AGENT] ── ENTITY GROUP 2 ─────────────────────────────────────────────
    # Purpose : A second sub-group of the same entity type, differentiated by
    #           era/category. Here: TNG characters. For car racing this could
    #           be "modern era drivers"; for cooking: "pastry chefs".
    # Action  : Replace list and templates, or merge into Entity Group 1 if
    #           your topic doesn't have meaningful sub-categories.
    # ──────────────────────────────────────────────────────────────────────
    # Characters - TNG
    tng_characters = [
        "Jean-Luc Picard", "William Riker", "Data", "Worf", "Geordi La Forge",
        "Deanna Troi", "Beverly Crusher", "Wesley Crusher", "Tasha Yar", "Guinan",
        "Q", "Lwaxana Troi", "Alexander Rozhenko", "Reginald Barclay", "Miles O'Brien",
        "Keiko O'Brien", "Ro Laren", "Ensign Sito Jaxa", "Alyssa Ogawa", "Mott"
    ]

    for char in tng_characters:
        questions.append(f'Who is {char} in Star Trek: The Next Generation?')
        questions.append(f'What is {char}\'s position or role?')
        if "Picard" in char:
            questions.append(f'What is {char}\'s favorite beverage?')
            questions.append(f'What is {char}\'s catchphrase?')
        if "Data" in char:
            questions.append(f'What is {char}\'s quest?')
            questions.append(f'What is {char}\'s cat\'s name?')

    # [AGENT] ── ENTITY GROUPS 3-N ────────────────────────────────────────
    # Repeat the same pattern for as many sub-groups as your topic needs.
    # Each group should have a distinct list + 1-3 templated questions.
    # Star Trek uses series (DS9, VOY, ENT, DIS, PIC, LD) as sub-groups.
    # Car racing could use: F1 teams, NASCAR drivers, rally drivers, etc.
    # Delete any groups that have no equivalent in your topic.
    # ──────────────────────────────────────────────────────────────────────
    # Characters - DS9
    ds9_characters = [
        "Benjamin Sisko", "Kira Nerys", "Odo", "Quark", "Jadzia Dax", "Ezri Dax",
        "Julian Bashir", "Miles O'Brien", "Worf", "Jake Sisko", "Kasidy Yates",
        "Gul Dukat", "Weyoun", "Damar", "Garak", "Kai Winn", "Vedek Bareil",
        "Rom", "Nog", "Leeta", "Zek", "Ishka", "Brunt"
    ]

    for char in ds9_characters:
        questions.append(f'What is {char}\'s role in Deep Space Nine?')
        if "Sisko" in char:
            questions.append(f'What is {char}\'s connection to the Prophets?')
            questions.append(f'What is {char}\'s rank?')
        if "Dax" in char:
            questions.append(f'What is the Dax symbiont?')
            questions.append(f'How many hosts has the Dax symbiont had?')

    # Characters - VOY
    voy_characters = [
        "Kathryn Janeway", "Chakotay", "Tuvok", "Tom Paris", "B'Elanna Torres",
        "Harry Kim", "Seven of Nine", "The Doctor", "Neelix", "Kes",
        "Seska", "Lon Suder", "Icheb", "Naomi Wildman", "Q"
    ]

    for char in voy_characters:
        questions.append(f'Who is {char} in Star Trek: Voyager?')
        if "Janeway" in char:
            questions.append(f'What was {char}\'s mission?')
            questions.append(f'What is {char}\'s rank?')
        if "Seven" in char or "Nine" in char:
            questions.append(f'What is {char}\'s Borg designation?')
            questions.append(f'What is {char}\'s real name?')

    # Characters - ENT
    ent_characters = [
        "Jonathan Archer", "T'Pol", "Trip Tucker", "Malcolm Reed", "Hoshi Sato",
        "Travis Mayweather", "Phlox", "Shran", "Soval", "Daniels"
    ]

    for char in ent_characters:
        questions.append(f'What is {char}\'s role in Enterprise?')
        if "Archer" in char:
            questions.append(f'What is {char}\'s ship?')
            questions.append(f'What is {char}\'s rank?')

    # Characters - DIS
    dis_characters = [
        "Michael Burnham", "Saru", "Paul Stamets", "Sylvia Tilly", "Ash Tyler",
        "Philippa Georgiou", "Gabriel Lorca", "Christopher Pike", "Spock", "Adira Tal",
        "Gray Tal", "Jett Reno", "Keyla Detmer", "Joann Owosekun", "Ronald Altman Bryce"
    ]

    for char in dis_characters:
        questions.append(f'Who is {char} in Star Trek: Discovery?')
        if "Burnham" in char:
            questions.append(f'What is {char}\'s relationship to Spock?')
            questions.append(f'What is {char}\'s rank?')

    # Characters - PIC
    pic_characters = [
        "Jean-Luc Picard", "Raffi Musiker", "Cristobal Rios", "Agnes Jurati", "Elnor",
        "Soji Asha", "Dahj Asha", "Narek", "Narissa", "Laris", "Zhaban", "Q"
    ]

    for char in pic_characters:
        questions.append(f'What is {char}\'s role in Star Trek: Picard?')

    # Characters - LD
    ld_characters = [
        "Brad Boimler", "Beckett Mariner", "D'Vana Tendi", "Sam Rutherford", "Jack Ransom",
        "Carol Freeman", "Shaxs", "T'Ana", "Andy Billups", "Miguel Castro"
    ]

    for char in ld_characters:
        questions.append(f'Who is {char} in Star Trek: Lower Decks?')

    # Characters - SNW
    snw_characters = [
        "Christopher Pike", "Spock", "Number One", "La'an Noonien-Singh", "Erica Ortegas",
        "M'Benga", "Nurse Chapel", "Hemmer", "Uhura", "Sam Kirk",
        "T'Pring", "Khan Noonien-Singh", "Jess Bush", "Anson Mount"
    ]

    for char in snw_characters:
        questions.append(f'Who is {char} in Star Trek: Strange New Worlds?')
        questions.append(f'What is {char}\'s role in Strange New Worlds?')

    # Characters - Prodigy
    prodigy_characters = [
        "Dal R'El", "Gwyndala", "Jankom Pog", "Zero", "Rok-Tahk", "Murf",
        "Hologram Janeway", "Chakotay", "The Diviner", "Drednok", "Asencia"
    ]

    for char in prodigy_characters:
        questions.append(f'Who is {char} in Star Trek: Prodigy?')

    # Characters - TAS (Animated Series)
    tas_characters = [
        "James T. Kirk", "Spock", "Leonard McCoy", "Arex", "M'Ress",
        "Montgomery Scott", "Uhura", "Sulu", "Chekov"
    ]

    for char in tas_characters:
        questions.append(f'Who is {char} in Star Trek: The Animated Series?')

    # [AGENT] ── NAMED OBJECTS / EQUIPMENT ────────────────────────────────
    # Purpose : The iconic named objects associated with the topic.
    #           Star Trek uses spaceships. For car racing: iconic race cars
    #           (e.g. Ferrari F2004, McLaren MP4/4). For cooking: famous
    #           restaurants or signature dishes. For football: famous stadiums.
    # Pattern : 2-3 questions per object, focusing on identity and significance.
    # Action  : Replace lists and question templates.
    # ──────────────────────────────────────────────────────────────────────
    # Spaceships - Enterprise variants
    enterprise_ships = [
        "USS Enterprise NCC-1701", "USS Enterprise NCC-1701-A", "USS Enterprise NCC-1701-B",
        "USS Enterprise NCC-1701-C", "USS Enterprise NCC-1701-D", "USS Enterprise NCC-1701-E",
        "USS Enterprise NCC-1701-F", "USS Enterprise NX-01"
    ]

    for ship in enterprise_ships:
        questions.append(f'Who commanded the {ship}?')
        questions.append(f'What class is the {ship}?')
        questions.append(f'What is the {ship} known for?')

    # [AGENT] Additional named objects sub-group. Keep or remove as needed.
    # Other notable ships
    other_ships = [
        "USS Voyager NCC-74656", "USS Defiant NX-74205", "USS Discovery NCC-1031",
        "USS Cerritos NCC-75567", "USS Reliant NCC-1864", "USS Excelsior NX-2000",
        "USS Enterprise NCC-1701-J", "USS Stargazer NCC-2893", "USS Titan NCC-80102",
        "USS Prometheus NX-59650", "USS Equinox NCC-72381", "USS Saratoga NCC-1887",
        "USS Pasteur NCC-58925"
    ]

    for ship in other_ships:
        questions.append(f'What is the {ship}?')
        questions.append(f'What series featured the {ship}?')

    # [AGENT] Faction-specific objects sub-group (here: Klingon ships).
    #         For car racing: objects associated with a specific constructor
    #         or team (e.g. Red Bull cars, Mercedes cars). Remove if not needed.
    # Klingon ships
    klingon_ships = [
        "IKS Bortas", "IKS Rotarran", "IKS Kronos One", "IKS Negh'Var", "IKS D'Ghor",
        "Bird of Prey", "K't'inga class", "Vor'cha class", "D7 class"
    ]

    for ship in klingon_ships:
        questions.append(f'What is a {ship} in Star Trek?')
        questions.append(f'Which species uses the {ship}?')

    # Romulan ships
    romulan_ships = [
        "Romulan Bird of Prey", "Romulan Warbird", "D'deridex class", "Valdore type",
        "Scimitar", "Narada"
    ]

    for ship in romulan_ships:
        questions.append(f'What is a {ship}?')
        questions.append(f'What is the {ship} known for?')

    # Borg ships
    borg_ships = [
        "Borg Cube", "Borg Sphere", "Borg Tactical Cube", "Borg Queen's Vessel"
    ]

    for ship in borg_ships:
        questions.append(f'What is a {ship}?')
        questions.append(f'What is the purpose of a {ship}?')

    # [AGENT] ── LOCATIONS ────────────────────────────────────────────────
    # Purpose : Named places significant to the topic.
    #           Star Trek uses fictional planets. For car racing: famous
    #           circuits (Monza, Silverstone, Monaco). For cooking: culinary
    #           regions or cities (Lyon, Tokyo, New Orleans). For football:
    #           famous grounds or cities associated with the sport.
    # Pattern : 3 questions per location covering identity, inhabitants/users,
    #           and a notable event that occurred there.
    # Action  : Replace list and question templates.
    # ──────────────────────────────────────────────────────────────────────
    # Planets and locations
    planets = [
        "Vulcan", "Romulus", "Remus", "Qo'noS", "Cardassia Prime", "Bajor",
        "Earth", "Andoria", "Tellar Prime", "Trill", "Betazed", "Ferenginar",
        "Risa", "Talos IV", "Genesis Planet", "Veridian III", "Ba'ku", "New Earth",
        "Terralysium", "Kaminar", "Xahea", "Kwejian", "Kataan", "Aldea", "Iconia",
        "Organia", "Eminiar VII", "Vendikar", "Archer IV", "P'Jem", "Rigel X",
        "Denobula", "Coridan", "Deneva", "Tarsus IV", "Janus VI", "Miri's World",
        "Gamma Hydra IV", "Beta III", "Omega IV", "Terra Nova", "Ceti Alpha V",
        "Regula I", "Nimbus III", "Sha Ka Ree"
    ]

    for planet in planets:
        questions.append(f'What is {planet} in Star Trek?')
        questions.append(f'Which species is from {planet}?')
        questions.append(f'What happened on {planet}?')

    # [AGENT] Secondary locations sub-group (here: space stations).
    #         For car racing: pit lane facilities, race control buildings.
    #         Remove if your topic has no meaningful secondary location type.
    # Space stations
    stations = [
        "Deep Space 9", "Starbase 1", "Starbase 11", "Starbase 80", "Yorktown Station",
        "K-7", "Regula I", "Empok Nor", "Terok Nor", "Starbase 375"
    ]

    for station in stations:
        questions.append(f'What is {station}?')
        questions.append(f'What series featured {station}?')

    # [AGENT] ── TECHNICAL / DOMAIN CONCEPTS ─────────────────────────────
    # Purpose : The specialised vocabulary, tools, and mechanisms of the topic.
    #           Star Trek uses in-universe technology. For car racing: engine
    #           types, aero components, DRS, KERS, pitstop equipment. For
    #           cooking: techniques (sous vide, emulsification, tempering).
    # Pattern : 3 questions per concept covering function, purpose, and origin.
    # Action  : Replace list and question templates. Keep 3 questions per item
    #           to maintain question density.
    # ──────────────────────────────────────────────────────────────────────
    # Technology
    tech_concepts = [
        "transporter", "warp drive", "phaser", "photon torpedo", "quantum torpedo",
        "tricorder", "medical tricorder", "replicator", "holodeck", "warp core",
        "deflector array", "warp nacelles", "impulse drive", "shields", "cloaking device",
        "universal translator", "communicator", "PADD", "bio-neural gel packs",
        "isolinear chips", "duotronic systems", "spore drive", "mycelial network",
        "temporal drive", "quantum slipstream", "transwarp", "Borg transwarp conduits",
        "warp field", "subspace", "tachyon", "chroniton", "chronometric particles"
    ]

    for tech in tech_concepts:
        questions.append(f'How does a {tech} work in Star Trek?')
        questions.append(f'What is the purpose of a {tech}?')
        questions.append(f'Which species invented the {tech}?')

    # [AGENT] ── FACTIONS / GROUPS / SPECIES ─────────────────────────────
    # Purpose : The distinct groups, teams, or factions within the domain.
    #           Star Trek uses alien species. For car racing: constructors /
    #           teams (Ferrari, Mercedes, Red Bull, McLaren). For cooking:
    #           culinary schools or traditions (French, Japanese, Peruvian).
    #           For football: clubs or national federations.
    # Pattern : 4 questions per group covering reputation, characteristics,
    #           home base, and relationship to a governing body.
    # Action  : Replace list and reword templates for the new topic.
    # ──────────────────────────────────────────────────────────────────────
    # Species
    species_list = [
        "Vulcan", "Romulan", "Klingon", "Borg", "Cardassian", "Bajoran", "Ferengi",
        "Trill", "Betazoid", "Andorian", "Tellarite", "Bolian", "Caitian", "Denobulan",
        "Vorta", "Jem'Hadar", "Founders", "Species 8472", "Talaxian", "Ocampa",
        "Kazon", "Vidiian", "Hirogen", "Malon", "Breen", "Son'a", "Ba'ku",
        "Xindi", "Suliban", "Kelpien", "Orion", "Gorn", "Tholian", "Sheliak",
        "Tamarian", "Pakled", "Tzenkethi", "Hazari", "Vaadwaur", "Krenim"
    ]

    for species in species_list:
        questions.append(f'What are the {species} known for?')
        questions.append(f'What are the characteristics of the {species}?')
        questions.append(f'What is the {species} homeworld?')
        questions.append(f'What is the relationship between the {species} and the Federation?')

    # [AGENT] ── GOVERNING BODIES / ORGANISATIONS ────────────────────────
    # Purpose : Official bodies, leagues, associations, and institutions.
    #           Star Trek uses Starfleet, the Federation, etc. For car racing:
    #           FIA, Formula 1 Group, ACO, NASCAR, IndyCar. For cooking:
    #           Michelin Guide, James Beard Foundation, culinary institutes.
    # Pattern : 3 questions per organisation covering identity, purpose, leader.
    # Action  : Replace list and templates.
    # ──────────────────────────────────────────────────────────────────────
    # Organizations
    organizations = [
        "United Federation of Planets", "Starfleet", "Section 31", "Tal Shiar",
        "Obsidian Order", "Klingon High Council", "Bajoran Provisional Government",
        "Ferengi Commerce Authority", "Dominion", "Borg Collective", "Q Continuum",
        "Temporal Integrity Commission", "Department of Temporal Investigations",
        "Vulcan Science Academy", "Starfleet Academy", "Starfleet Command",
        "Starfleet Medical", "Starfleet Intelligence", "Maquis", "Terran Empire"
    ]

    for org in organizations:
        questions.append(f'What is the {org}?')
        questions.append(f'What is the purpose of the {org}?')
        questions.append(f'Who leads the {org}?')

    # [AGENT] ── NOTABLE EVENTS ───────────────────────────────────────────
    # Purpose : Landmark moments, competitions, incidents, or seasons.
    #           Star Trek uses battles and wars. For car racing: famous races
    #           or seasons (1976 season, 1988 Monaco GP, Senna vs Prost).
    #           For cooking: famous cook-offs, restaurant openings, or scandals.
    # Pattern : 3 questions per event — what happened, when, who was involved.
    # Action  : Replace list and keep the 3-question pattern.
    # ──────────────────────────────────────────────────────────────────────
    # Events and battles
    events = [
        "Battle of Wolf 359", "Battle of Sector 001", "Dominion War", "Klingon-Federation War",
        "Romulan War", "Battle of Cardassia", "Battle of Chin'toka", "Battle of AR-558",
        "First Contact Day", "The Burn", "Time War", "Earth-Romulan War",
        "Xindi attack on Earth", "Khitomer Massacre", "Narendra III", "Praxis explosion",
        "Destruction of Romulus", "Attack on Mars", "Battle of the Binary Stars",
        "Battle of the Mutara Nebula", "Genesis Incident", "Veridian III incident"
    ]

    for event in events:
        questions.append(f'What happened during the {event}?')
        questions.append(f'When did the {event} occur?')
        questions.append(f'Who was involved in the {event}?')

    # [AGENT] ── CANONICAL WORKS / EPISODES / SEASONS ────────────────────
    # Purpose : The individual works, episodes, seasons, or editions that make
    #           up the canon of the topic. Star Trek uses episode titles.
    #           For car racing: individual Grand Prix races or championship
    #           seasons. For cooking: cookbook titles or TV show episodes.
    #           For football: famous matches or cup finals.
    # Pattern : 2 questions per work — what happens / what is it about.
    # Action  : Replace episode lists and adjust templates for the new topic.
    #           Keep multiple sub-groups if the topic spans multiple eras.
    # ──────────────────────────────────────────────────────────────────────
    # Episodes - TOS
    tos_episodes = [
        "The City on the Edge of Forever", "The Trouble with Tribbles", "Balance of Terror",
        "Mirror, Mirror", "Space Seed", "The Doomsday Machine", "Amok Time", "The Menagerie",
        "Journey to Babel", "The Enterprise Incident", "Arena", "The Corbomite Maneuver",
        "Where No Man Has Gone Before", "The Naked Time", "Shore Leave", "The Squire of Gothos"
    ]

    for ep in tos_episodes:
        questions.append(f'What happens in the Star Trek episode "{ep}"?')
        questions.append(f'What is "{ep}" about?')

    # Episodes - TNG
    tng_episodes = [
        "The Best of Both Worlds", "Yesterday's Enterprise", "The Inner Light", "Chain of Command",
        "All Good Things...", "Darmok", "The Measure of a Man", "Q Who", "The Offspring",
        "Sarek", "Unification", "Redemption", "Cause and Effect", "Time's Arrow", "Relics",
        "Tapestry", "Frame of Mind", "The Pegasus", "Lower Decks", "The First Duty"
    ]

    for ep in tng_episodes:
        questions.append(f'What happens in The Next Generation episode "{ep}"?')
        questions.append(f'What is "{ep}" about?')

    # Episodes - DS9
    ds9_episodes = [
        "The Emissary", "Duet", "In the Hands of the Prophets", "The Jem'Hadar",
        "The Way of the Warrior", "Trials and Tribble-ations", "In Purgatory's Shadow",
        "By Inferno's Light", "Call to Arms", "Sacrifice of Angels", "In the Pale Moonlight",
        "The Siege of AR-558", "It's Only a Paper Moon", "Far Beyond the Stars", "The Visitor"
    ]

    for ep in ds9_episodes:
        questions.append(f'What happens in Deep Space Nine episode "{ep}"?')
        questions.append(f'What is "{ep}" about?')

    # Episodes - VOY
    voy_episodes = [
        "Caretaker", "Scorpion", "Year of Hell", "Timeless", "Blink of an Eye",
        "The Killing Game", "Dark Frontier", "Unimatrix Zero", "Endgame", "Living Witness",
        "Tuvix", "Threshold", "Message in a Bottle", "Hope and Fear", "Relativity"
    ]

    for ep in voy_episodes:
        questions.append(f'What happens in Voyager episode "{ep}"?')
        questions.append(f'What is "{ep}" about?')

    # [AGENT] ── MAJOR / FEATURE WORKS ───────────────────────────────────
    # Purpose : The high-profile, standalone centrepieces of the topic.
    #           Star Trek uses theatrical films. For car racing: the most
    #           prestigious races (Monaco, Le Mans, Indy 500, Daytona 500).
    #           For cooking: Michelin 3-star restaurants or landmark cookbooks.
    # Pattern : 3 questions per work — plot/overview, key moment, antagonist
    #           or main challenge. Adapt the third question template if
    #           "villain" is not relevant (e.g. "Who won?", "What record was set?").
    # Action  : Replace list and question templates.
    # ──────────────────────────────────────────────────────────────────────
    # Films
    films = [
        "Star Trek: The Motion Picture", "Star Trek II: The Wrath of Khan",
        "Star Trek III: The Search for Spock", "Star Trek IV: The Voyage Home",
        "Star Trek V: The Final Frontier", "Star Trek VI: The Undiscovered Country",
        "Star Trek: Generations", "Star Trek: First Contact", "Star Trek: Insurrection",
        "Star Trek: Nemesis", "Star Trek (2009)", "Star Trek Into Darkness",
        "Star Trek Beyond"
    ]

    for film in films:
        questions.append(f'What is the plot of {film}?')
        questions.append(f'What happens in {film}?')
        questions.append(f'Who is the villain in {film}?')

    # [AGENT] ── CONCEPTS, RULES & PHILOSOPHIES ──────────────────────────
    # Purpose : The abstract principles, rules, or philosophies central to
    #           the topic. Star Trek uses the Prime Directive, IDIC, etc.
    #           For car racing: "racing line", parc fermé rules, Gentleman's
    #           Agreement, undercut strategy. For cooking: mise en place,
    #           flavour pairing theory, knife skills philosophy.
    # Pattern : 2 questions per concept — definition and meaning/significance.
    # Action  : Replace list and templates.
    # ──────────────────────────────────────────────────────────────────────
    # Concepts and philosophies
    concepts = [
        "Prime Directive", "Temporal Prime Directive", "IDIC", "Logic", "Kol-ut-shan",
        "Klingon honor", "Ferengi Rules of Acquisition", "Bajoran Prophets", "Pah-wraiths",
        "Wormhole", "Bajoran wormhole", "Temporal mechanics", "Parallel universe",
        "Mirror Universe", "Alternate timeline", "Butterfly effect", "Grandfather paradox"
    ]

    for concept in concepts:
        questions.append(f'What is the {concept} in Star Trek?')
        questions.append(f'What does the {concept} mean?')

    # [AGENT] ── FAMOUS QUOTES & CATCHPHRASES ────────────────────────────
    # Purpose : Iconic lines or phrases strongly associated with the topic.
    #           Star Trek uses character catchphrases. For car racing: famous
    #           team radio messages, podium quotes, commentator lines.
    #           For cooking: famous chef quotes ("Season boldly", etc.).
    # Pattern : 3 questions per quote — who said it, context, and where it
    #           appears. If quotes are less prominent in your topic, reduce
    #           to 2 questions or merge this section with Concepts above.
    # Action  : Replace list and templates.
    # ──────────────────────────────────────────────────────────────────────
    # Quotes and phrases
    quotes = [
        "Live long and prosper", "Make it so", "Engage", "Fascinating", "He's dead, Jim",
        "I'm a doctor, not a...", "Beam me up, Scotty", "To boldly go where no one has gone before",
        "Resistance is futile", "The needs of the many", "KHAN!", "Darmok and Jalad at Tanagra",
        "There are four lights", "It is a good day to die", "Qapla'", "Today is a good day to die",
        "The first duty of every Starfleet officer", "Infinite diversity in infinite combinations"
    ]

    for quote in quotes:
        questions.append(f'Who said "{quote}" in Star Trek?')
        questions.append(f'What is the context of "{quote}"?')
        questions.append(f'What episode or film features "{quote}"?')

    # [AGENT] ── SPECIALISED TOOLS / INSTRUMENTS ─────────────────────────
    # Purpose : The specific tools, instruments, or apparatus used within the
    #           domain. Star Trek uses medical/scientific devices. For car
    #           racing: tyre compounds, telemetry systems, HANS device, DRS
    #           flap. For cooking: specific knives, thermometers, mandoline.
    # Pattern : 2 questions per item — what it is and how it is used.
    # Action  : Replace list and templates. Rename the variable appropriately.
    # ──────────────────────────────────────────────────────────────────────
    # Medical and science
    medical = [
        "hypospray", "dermal regenerator", "cortical stimulator", "neurocortical monitor",
        "surgical laser", "bone knitter", "cardiac stimulator", "synthehol", "anesthizine",
        "cordrazine", "tri-ox compound", "polywater", "pon farr", "katra", "mind meld"
    ]

    for med in medical:
        questions.append(f'What is a {med} in Star Trek?')
        questions.append(f'How is a {med} used?')

    # [AGENT] ── COMPETITIVE / OFFENSIVE ELEMENTS ────────────────────────
    # Purpose : Items or tactics associated with competition or conflict.
    #           Star Trek uses weapons. For car racing: aggressive driving
    #           manoeuvres (the "dive bomb", "blocking move"), penalty types.
    #           For cooking: knife types used in competition. For football:
    #           set pieces, pressing styles.
    #           Remove this section entirely if it has no equivalent.
    # Action  : Replace list, rename variable, update templates.
    # ──────────────────────────────────────────────────────────────────────
    # Weapons
    weapons = [
        "phaser", "disruptor", "bat'leth", "d'k tahg", "mek'leth", "lirpa", "ahn-woon",
        "Klingon painstik", "Vulcan lirpa", "Andorian Ushaan-tor", "Borg cutting beam"
    ]

    for weapon in weapons:
        questions.append(f'What is a {weapon}?')
        questions.append(f'Which species uses the {weapon}?')

    # [AGENT] ── CULTURAL ARTEFACTS / CONSUMABLES ────────────────────────
    # Purpose : The food, drink, rituals, or cultural objects associated with
    #           the topic's world. Star Trek uses in-universe foods and drinks.
    #           For car racing: celebratory champagne, energy drinks, team
    #           merchandise. For football: stadium foods.
    #           Remove or repurpose if not relevant to the new topic.
    # Action  : Replace list and rename variable if needed.
    # ──────────────────────────────────────────────────────────────────────
    # Food and drink
    food_items = [
        "gagh", "raktajino", "hasperat", "plomeek soup", "Vulcan port", "Saurian brandy",
        "Romulan ale", "prune juice", "root beer", "Earl Grey tea", "chocolate", "peanut butter",
        "jambalaya", "pizza", "warp core breach", "Chateau Picard"
    ]

    for item in food_items:
        questions.append(f'What is {item} in Star Trek?')
        questions.append(f'Who is known for consuming {item}?')

    # [AGENT] ── TOPIC-SPECIFIC MECHANICS ────────────────────────────────
    # Purpose : A specialised sub-domain or advanced mechanic unique to the
    #           topic. Star Trek uses time travel. For car racing: tyre
    #           strategy, fuel load effects, safety car periods. For cooking:
    #           fermentation science, Maillard reaction. For football:
    #           VAR technology, offside mechanics.
    #           Add or remove this section based on whether your topic has a
    #           natural "advanced mechanics" sub-domain worth covering.
    # Action  : Replace list and templates, rename variable.
    # ──────────────────────────────────────────────────────────────────────
    # Time travel
    time_travel = [
        "time travel", "temporal causality loop", "temporal anomaly", "temporal incursion",
        "temporal prime directive", "Department of Temporal Investigations", "temporal agent",
        "temporal cold war", "butterfly effect", "grandfather paradox"
    ]

    for tt in time_travel:
        questions.append(f'How does {tt} work in Star Trek?')
        questions.append(f'What is the {tt}?')

    # [AGENT] ── HAND-CRAFTED ENTITY QUESTIONS ───────────────────────────
    # Purpose : Specific, hand-written questions about the most famous
    #           individuals that loop-generated questions might miss.
    #           Star Trek covers parentage, ranks, pet names, etc.
    #           For car racing: "How many F1 titles did Senna win?",
    #           "What nationality is Max Verstappen?". Keep this list to
    #           cover gaps left by the loop-generated sections above.
    # Action  : Replace with hand-crafted questions for the new topic's
    #           most important entities.
    # ──────────────────────────────────────────────────────────────────────
    # Additional character questions
    questions.extend([
        "What is Spock's human mother's name?", "What is Spock's father's name?",
        "What is Kirk's middle name?", "What is Picard's first name?",
        "What is Data's brother's name?", "What is Worf's son's name?",
        "What is Janeway's first name?", "What is Sisko's first name?",
        "What is O'Brien's first name?", "What is Bashir's first name?",
        "What is Dax's first host?", "What is the name of Seven of Nine's parents?",
        "What is T'Pol's rank?", "What is Archer's dog's name?",
        "What is Burnham's rank?", "What is Saru's species?"
    ])

    # [AGENT] ── HAND-CRAFTED OBJECT QUESTIONS ───────────────────────────
    # Purpose : Specific questions about the topic's key objects not covered
    #           by loop generation. Star Trek covers ship classes and registry
    #           numbers. For car racing: engine specs, lap records, chassis
    #           numbers. Remove if the loop sections above are sufficient.
    # Action  : Replace or remove.
    # ──────────────────────────────────────────────────────────────────────
    # Additional ship questions
    questions.extend([
        "What is the registry number of the USS Voyager?", "What is the registry number of the USS Defiant?",
        "What class is the USS Voyager?", "What class is the USS Defiant?",
        "What is the maximum warp speed of the Enterprise-D?", "What is a Galaxy class starship?",
        "What is a Sovereign class starship?", "What is an Intrepid class starship?",
        "What is a Defiant class starship?", "What is a Constitution class starship?",
        "What is an NX class starship?", "What is a Crossfield class starship?",
        "What is a California class starship?", "What is a Miranda class starship?",
        "What is a Nebula class starship?", "What is an Excelsior class starship?"
    ])

    # [AGENT] ── HAND-CRAFTED TECHNICAL QUESTIONS ────────────────────────
    # Purpose : Specific technical questions not covered by loop generation.
    #           Star Trek covers warp speeds, subspace, etc. For car racing:
    #           maximum RPM limits, DRS activation zones, fuel flow rates.
    #           Remove if the loop sections above are sufficient.
    # Action  : Replace or remove.
    # ──────────────────────────────────────────────────────────────────────
    # Additional technology questions
    questions.extend([
        "What is the maximum warp speed in Star Trek?", "What is warp 10?",
        "What happens at warp 10?", "What is transwarp?",
        "What is a warp core breach?", "What is a matter-antimatter reaction?",
        "What is dilithium?", "What is a warp field?",
        "What is subspace?", "What is a subspace communication?",
        "What is a subspace anomaly?", "What is a tachyon?",
        "What is a chroniton?", "What is a chronometric particle?",
        "What is a quantum singularity?", "What is a graviton?",
        "What is a polaron?", "What is an isokinetic cannon?"
    ])

    # [AGENT] ── REAL-WORLD STAR TREK (actors, creators, production) ──────
    # Purpose : Questions about Star Trek as a real-world franchise — actors,
    #           writers, directors, conventions, merchandise. These are clearly
    #           Trek-related but would be missed by in-universe question templates.
    # Action  : Replace or extend with equivalent real-world facts for your topic.
    # ──────────────────────────────────────────────────────────────────────
    actors_roles = [
        ("William Shatner", "Captain Kirk"),
        ("Leonard Nimoy", "Spock"),
        ("DeForest Kelley", "Dr McCoy"),
        ("James Doohan", "Scotty"),
        ("Nichelle Nichols", "Uhura"),
        ("George Takei", "Sulu"),
        ("Walter Koenig", "Chekov"),
        ("Patrick Stewart", "Captain Picard"),
        ("Brent Spiner", "Data"),
        ("LeVar Burton", "Geordi La Forge"),
        ("Michael Dorn", "Worf"),
        ("Marina Sirtis", "Deanna Troi"),
        ("Gates McFadden", "Dr Crusher"),
        ("Jonathan Frakes", "Riker"),
        ("Avery Brooks", "Benjamin Sisko"),
        ("Nana Visitor", "Kira Nerys"),
        ("René Auberjonois", "Odo"),
        ("Armin Shimerman", "Quark"),
        ("Terry Farrell", "Jadzia Dax"),
        ("Kate Mulgrew", "Captain Janeway"),
        ("Robert Beltran", "Chakotay"),
        ("Jeri Ryan", "Seven of Nine"),
        ("Robert Picardo", "The Doctor"),
        ("Scott Bakula", "Jonathan Archer"),
        ("Sonequa Martin-Green", "Michael Burnham"),
        ("Doug Jones", "Saru"),
        ("Anthony Rapp", "Paul Stamets"),
        ("Anson Mount", "Christopher Pike"),
        ("Ethan Peck", "Spock in Strange New Worlds"),
        ("Michelle Yeoh", "Philippa Georgiou"),
    ]

    for actor, role in actors_roles:
        questions.append(f'Who does {actor} play in Star Trek?')
        questions.append(f'Which Star Trek actor plays {role}?')
        questions.append(f'What is {actor} known for in Star Trek?')

    questions.extend([
        "Who created Star Trek?",
        "Who is Gene Roddenberry?",
        "When did the original Star Trek series first air?",
        "What year did Star Trek: The Next Generation premiere?",
        "What network originally aired Star Trek?",
        "What is Star Trek's influence on modern technology?",
        "How many Star Trek series have been made?",
        "What is the Star Trek expanded universe?",
        "What are the Star Trek novels?",
        "Who wrote the most Star Trek episodes?",
        "What is Majel Barrett's connection to Star Trek?",
        "What is D.C. Fontana's contribution to Star Trek?",
        "What is the significance of Gene L. Coon to Star Trek?",
        "What is Nichelle Nichols' cultural impact?",
        "How did Star Trek influence NASA?",
        "What is the Star Trek franchise worth?",
        "What is Star Trek Online?",
        "What is the Star Trek roleplaying game?",
        "What are the FASA Star Trek games?",
        "What is the Star Trek Customizable Card Game?",
        "What is the USS Enterprise model kit?",
        "What happens at a Star Trek convention?",
        "What is Trekker vs Trekkie?",
        "What is the Vulcan salute?",
        "Who invented the Vulcan salute?",
        "What is stardate format?",
        "What does NCC stand for in Star Trek?",
        "What does USS stand for in Star Trek?",
        "What is the Star Trek theme music called?",
        "Who composed the Star Trek theme?",
        "What is the difference between Star Trek and Star Wars?",
        "What Star Trek props are on display in museums?",
    ])

    # [AGENT] ── NATURAL-LANGUAGE PHRASINGS ───────────────────────────────
    # Purpose : Non-question inputs that mirror how real users actually write.
    #           Without these, the model may key on the "?" character and fail
    #           on statements, commands, and conversational inputs.
    # Action  : Replace with equivalent natural-language forms for your topic.
    # ──────────────────────────────────────────────────────────────────────
    questions.extend([
        "Tell me about the Prime Directive",
        "Explain how warp drive works",
        "I want to know about the Borg",
        "Give me a summary of Deep Space Nine",
        "Describe the Klingon Empire",
        "I'm curious about Vulcan culture",
        "Tell me everything about Captain Picard",
        "Explain the difference between Romulans and Vulcans",
        "I'd like to understand the Dominion War",
        "Can you summarise the plot of The Wrath of Khan?",
        "Talk me through how transporters work",
        "Give me some background on Seven of Nine",
        "Describe what happens in The Inner Light episode",
        "I want to learn about Starfleet Academy",
        "Explain Pon Farr",
        "Tell me about the Mirror Universe",
        "Walk me through the Dominion War timeline",
        "Give me the history of the Cardassian occupation of Bajor",
        "I'm interested in how the holodeck works",
        "Explain what a Trill symbiont is",
        "Tell me about Strange New Worlds season 2",
        "I want to know more about Star Trek: Prodigy",
        "Describe the animated Star Trek series",
        "Give me background on Gene Roddenberry",
        "Tell me about the filming locations for Star Trek",
        "Explain the Kelvin timeline",
        "What's the deal with Q?",
        "Who are the Founders?",
        "What's a bat'leth?",
        "Why do Vulcans suppress emotion?",
    ])

    # [AGENT] ── FILLER QUESTIONS ─────────────────────────────────────────
    # Purpose : Fills up to MAX_RELATED if the sections above don't reach it.
    #           The three filler templates below use Star Trek-specific phrasing.
    # Action  : Replace the three f-string templates with generic but on-topic
    #           alternatives for the new domain. Examples for car racing:
    #             f'What happened at race number {filler_index % 500} of the season?'
    #             f'What is the lap record at circuit {filler_index % 30}?'
    #             f'What is rule {filler_index % 50} of the FIA sporting code?'
    # ──────────────────────────────────────────────────────────────────────
    filler_index = 0
    while len(questions) < MAX_RELATED:
        questions.append(f'What is episode {filler_index % 1000} of Star Trek about?')
        if len(questions) < MAX_RELATED:
            questions.append(f'What is stardate {40000 + filler_index}?')
        if len(questions) < MAX_RELATED:
            questions.append(f'What is the {filler_index % 50}th Rule of Acquisition?')
        filler_index += 1

    return questions[:MAX_RELATED]


# ---------------------------------------------------------------------------
# Unrelated questions
# ---------------------------------------------------------------------------

def generate_unrelated_questions():
    # [AGENT] ── UNRELATED QUESTIONS OVERVIEW ────────────────────────────
    # This function generates questions that are clearly NOT about the topic.
    # The content here (cities, animals, tech companies, musicians, etc.) is
    # broadly off-topic for almost any specialist subject, so it can usually
    # be reused unchanged.
    #
    # EXCEPTION — check these before reusing:
    #   • sports list  — remove any sport that overlaps with the new topic
    #     (e.g. for car racing, remove "motorsport"-adjacent entries).
    #   • tech_companies — fine for almost all topics.
    #   • cities / countries — safe for all topics.
    #   • animals / plants — safe for all topics.
    #   • musicians / actors — safe for all topics.
    #
    # If the new topic IS food/cooking, also remove the foods list and the
    # hand-crafted cooking-adjacent questions from the general questions blocks.
    # ──────────────────────────────────────────────────────────────────────
    questions = []

    # [AGENT] Countries list — safe to reuse unchanged for any topic.
    countries = [
        "Canada", "Germany", "Japan", "Brazil", "Nigeria", "Sweden", "Thailand",
        "Ukraine", "Chile", "Finland", "Egypt", "Vietnam", "Kenya", "Norway",
        "Ireland", "Peru", "Malaysia", "Greece", "Portugal", "South Korea", "Netherlands", "Belgium",
        "Czech Republic", "Hungary", "Austria", "Switzerland", "Denmark", "Philippines",
        "Indonesia", "Turkey", "Saudi Arabia", "Argentina"
    ]

    cities = [
        "Osaka", "Munich", "Toronto", "Seoul", "Cairo", "Buenos Aires", "Stockholm",
        "Helsinki", "Bangkok", "Lisbon", "Warsaw", "Dublin", "Santiago", "Oslo",
        "Prague", "Budapest", "Vienna", "Zurich", "Copenhagen", "Manila", "Istanbul",
        "Jakarta", "Riyadh", "Lagos", "Kuala Lumpur", "Nairobi", "Accra", "Bogotá",
        "Lima", "Karachi", "Dhaka", "Ho Chi Minh City", "Casablanca", "Tunis",
        "Addis Ababa", "Dar es Salaam", "Yangon", "Kathmandu", "Colombo", "Tbilisi",
        "Yerevan", "Baku", "Tashkent", "Almaty", "Ulaanbaatar", "Phnom Penh",
        "Vientiane", "Dili", "Suva", "Apia", "Nuku'alofa"
    ]

    actors = [
        "Leonardo DiCaprio", "Scarlett Johansson", "Tom Hanks", "Charlize Theron",
        "Daniel Kaluuya", "Florence Pugh", "Rami Malek", "Viola Davis", "Mahershala Ali",
        "Chris Evans", "Zendaya", "Michael B. Jordan", "Emma Stone", "Denzel Washington",
        "Natalie Portman", "Cate Blanchett", "Anthony Hopkins", "Meryl Streep",
        "Brad Pitt", "Angelina Jolie", "Ryan Gosling", "Ana de Armas", "Pedro Pascal",
        "Sydney Sweeney", "Austin Butler", "Timothée Chalamet", "Margot Robbie",
        "Paul Mescal", "Barry Keoghan", "Andrew Garfield"
    ]

    musicians = [
        "Taylor Swift", "BTS", "Drake", "Billie Eilish", "Ed Sheeran", "Bad Bunny",
        "The Weeknd", "Rosé", "Shakira", "Coldplay", "Adele", "Bruno Mars", "Doja Cat",
        "Kendrick Lamar", "Dua Lipa", "Post Malone", "Ariana Grande", "Lady Gaga",
        "Justin Bieber", "Olivia Rodrigo", "Harry Styles", "Lizzo", "SZA",
        "Travis Scott", "Megan Thee Stallion", "Beyoncé", "Rihanna", "Jay-Z",
        "Eminem", "Kanye West", "Frank Ocean", "Tyler the Creator", "Lana Del Rey",
        "Arctic Monkeys", "Radiohead", "Tame Impala", "Fleetwood Mac", "The Beatles",
        "Led Zeppelin", "Pink Floyd", "David Bowie", "Elton John", "Queen"
    ]

    tech_companies = [
        "Apple", "Google", "Meta", "Amazon", "Samsung", "Sony", "Netflix", "Spotify",
        "Microsoft", "Tesla", "SpaceX", "OpenAI", "NVIDIA", "Intel", "AMD", "IBM", "Dell",
        "HP", "Cisco", "Oracle", "Salesforce", "Snapchat", "Twitter", "Reddit", "Pinterest", "Uber",
        "Airbnb", "Stripe", "Shopify", "Zoom", "Slack", "Palantir", "Snowflake",
        "MongoDB", "Cloudflare", "Twilio", "Block", "Robinhood", "Rivian", "Lucid"
    ]

    animals = [
        "koala", "kangaroo", "grizzly bear", "panda", "toucan", "narwhal", "axolotl",
        "quokka", "platypus", "maned wolf", "fossa", "sifaka", "capybara", "okapi", "aardvark",
        "tapir", "wolverine", "eland", "markhor", "dugong", "caracal", "serval",
        "binturong", "pangolin", "tenrec", "zorilla", "shoebill", "aye-aye", "tarsier",
        "slow loris", "clouded leopard", "snow leopard", "Amur leopard", "Siberian tiger",
        "blue-footed booby", "frigatebird", "kakapo", "kiwi", "cassowary", "emu",
        "wombat", "Tasmanian devil", "quoll", "numbat", "bilby", "echidna"
    ]

    plants = [
        "baobab", "dragon blood tree", "corpse flower", "joshua tree", "bottle tree",
        "weeping willow", "ginkgo biloba", "lotus", "cherry blossom", "banyan",
        "carnivorous pitcher plant", "saguaro cactus", "sensitive plant", "bristlecone pine",
        "giant sequoia", "venus flytrap", "rafflesia", "silk tree", "money tree",
        "ghost orchid", "lithops", "welwitschia", "mimosa pudica", "staghorn fern",
        "air plant", "california poppy", "bluebell", "bird of paradise", "titan arum",
        "black lotus", "rainbow eucalyptus", "quiver tree", "traveller's palm",
        "monkey puzzle tree", "dawn redwood", "wollemi pine", "sugarloaf pine"
    ]

    foods = [
        "ramen", "tacos", "paella", "kimchi", "sushi", "croissant", "borscht", "feijoada",
        "tagine", "pho", "hummus", "falafel", "samosa", "dumplings", "poke bowl", "ceviche",
        "pierogi", "gnocchi", "baklava", "churros", "bibimbap", "goulash", "arepas",
        "jambalaya", "moussaka", "fondue", "poutine"
    ]

    sports = [
        "cricket", "lacrosse", "curling", "bobsleigh", "badminton", "handball",
        "rugby sevens", "water polo", "gymnastics", "figure skating", "baseball",
        "American football", "ice hockey", "table tennis", "fencing", "archery",
        "triathlon", "cycling", "sailing", "equestrian", "shooting",
        "volleyball", "basketball", "association football", "tennis", "golf",
        "swimming", "diving", "rowing", "canoeing", "weightlifting",
        "boxing", "wrestling", "judo", "taekwondo", "karate",
        "ski jumping", "alpine skiing", "cross-country skiing", "biathlon",
        "speed skating", "short track speed skating", "luge", "skeleton",
        "modern pentathlon", "marathon", "sprinting", "long jump", "high jump",
        "pole vault", "javelin", "discus", "hammer throw", "shot put"
    ]

    for city in cities:
        questions.extend([
            f"What is the population of {city}?",
            f"How do you pronounce {city} correctly?",
            f"What time zone is {city} in?",
            f"What is the main airport in {city}?",
            f"What languages are spoken in {city}?",
            f"What is the crime rate like in {city}?",
            f"What is the public transit system like in {city}?",
            f"What are the best neighborhoods to stay in {city}?",
            f"What festivals happen in {city} annually?",
            f"What is the history of {city}?",
            f"What is the weather like in {city} throughout the year?",
            f"What are some famous landmarks in {city}?",
            f"What universities are located in {city}?",
            f"What is the cost of living in {city}?",
            f"What sports teams are based in {city}?",
            f"What is the local cuisine like in {city}?"
        ])

    for country in countries:
        questions.extend([
            f"What is the capital of {country}?",
            f"What is the official language of {country}?",
            f"What currency does {country} use?",
            f"What is the government structure of {country}?",
            f"What is the national dish of {country}?",
            f"What are the major exports of {country}?",
            f"What is the literacy rate in {country}?",
            f"What is the average salary in {country}?",
            f"What is the climate like in {country}?",
            f"What are the top tourist attractions in {country}?",
            f"What is the population of {country}?",
            f"What religions are practiced in {country}?",
            f"What is the history of {country}?",
            f"What festivals are celebrated in {country}?",
            f"What wildlife is native to {country}?"
        ])

    for actor in actors:
        questions.extend([
            f"What movies is {actor} known for?",
            f"Where was {actor} born?",
            f"What awards has {actor} won?",
            f"What is {actor}'s most recent film?",
            f"What is {actor}'s acting background?",
            f"How did {actor} get their start in acting?",
            f"What is {actor}'s net worth?",
            f"What is {actor}'s most acclaimed role?"
        ])

    questions.extend([
        "How do I reset my router password?",
        "What should I do if my phone won't charge?",
        "How can I recover deleted files on Windows?",
        "What is two-factor authentication?",
        "How do I clear cache in Chrome?",
        "Why is my Wi-Fi slow?",
        "How do I update my BIOS?",
        "What is a firewall?",
        "How do I check my IP address?",
        "What is phishing?",
        "How do I secure my smart home devices?",
        "Can I upgrade my laptop RAM myself?",
        "What is end-to-end encryption?",
        "How do I factory reset an Android phone?",
        "What is the difference between SSD and HDD?",
        "How do I free up space on my iPhone?",
        "What is a VPN and why should I use one?",
        "How do I transfer data from Android to iPhone?",
        "What is cloud storage?",
        "How do I set up parental controls on YouTube?",
        "What is the dark web?",
        "How do I enable cookies in my browser?",
        "What is machine learning?",
        "How do I create a strong password?",
        "What is the Internet of Things (IoT)?"
    ])

    questions.extend([
        "How do I unclog a sink naturally?",
        "What's the best way to store onions?",
        "How often should I water succulents?",
        "What is the legal driving age in Germany?",
        "How do I write a cover letter?",
        "What are the symptoms of dehydration?",
        "How do I stop snoring?",
        "What is the recommended daily intake of vitamin D?",
        "How much sleep do adults need?",
        "What is the average lifespan of a washing machine?",
        "How do I compost kitchen waste?",
        "What are the benefits of intermittent fasting?",
        "How do I train for a marathon?",
        "What is the proper way to lift weights?",
        "How do I meditate effectively?",
        "What are signs of burnout?",
        "How do I improve my credit score?",
        "What is compound interest?",
        "How do I negotiate a raise?",
        "What is the difference between stocks and bonds?",
        "How do I start a podcast?",
        "What are some tips for public speaking?",
        "How do I bake sourdough bread?",
        "What is the best way to learn a new language?",
        "How do I create a budget?"
    ])

    for food in foods:
        questions.extend([
            f"Where did {food} originate?",
            f"What ingredients are in {food}?",
            f"How do I make authentic {food} at home?",
            f"What is the history of {food}?",
            f"Is {food} gluten-free?",
            f"Can {food} be frozen?",
            f"What wine pairs well with {food}?",
            f"Is {food} vegan?",
            f"What spices are used in {food}?",
            f"Is {food} spicy by default?"
        ])

    for animal in animals:
        questions.extend([
            f"Where is the natural habitat of the {animal}?",
            f"Is the {animal} endangered?",
            f"What does a {animal} eat?",
            f"How long do {animal}s live?",
            f"Can {animal}s be domesticated?",
            f"What predators threaten the {animal}?",
            f"Is the {animal} nocturnal?",
            f"What is the scientific name for the {animal}?",
            f"Are {animal}s social animals?",
            f"Where can I see a wild {animal}?",
            f"How do {animal}s communicate?",
            f"What adaptations help the {animal} survive?"
        ])

    for plant in plants:
        questions.extend([
            f"Where does the {plant} grow naturally?",
            f"Is the {plant} poisonous?",
            f"How much sunlight does a {plant} need?",
            f"Can I grow {plant} indoors?",
            f"What soil type does {plant} prefer?",
            f"Is the {plant} drought-resistant?",
            f"Does the {plant} attract pollinators?",
            f"When does the {plant} bloom?",
            f"Is the {plant} invasive in some regions?",
            f"What animals depend on the {plant}?",
            f"What medicinal properties does the {plant} have?",
            f"How do I propagate a {plant}?"
        ])

    for musician in musicians:
        questions.extend([
            f"When was {musician} born?",
            f"Where is {musician} from?",
            f"What genre does {musician} perform?",
            f"What awards has {musician} won?",
            f"What is {musician}'s most popular song?",
            f"Has {musician} toured internationally?",
            f"Is {musician} active on social media?",
            f"What instruments does {musician} play?",
            f"Has {musician} collaborated with other artists?",
            f"Is {musician} involved in philanthropy?",
            f"What is {musician}'s latest album?",
            f"How did {musician} start their career?",
            f"What is {musician}'s best-selling album?",
            f"Has {musician} won a Grammy?",
            f"What is {musician}'s vocal range?",
            f"What is {musician}'s most streamed song on Spotify?"
        ])

    questions.extend([
        "What causes the northern lights?",
        "How do black holes form?",
        "What is quantum entanglement?",
        "Why is the sky blue?",
        "How do tornadoes develop?",
        "What is photosynthesis?",
        "How do vaccines work?",
        "What is CRISPR gene editing?",
        "Why do we dream?",
        "How do glaciers shape landscapes?",
        "What is dark matter?",
        "How do stars die?",
        "What is continental drift?",
        "Why do cats purr?",
        "How do bees communicate?",
        "What is the water cycle?",
        "How do earthquakes happen?",
        "What is the greenhouse effect?",
        "Why do leaves change color in autumn?",
        "How do birds navigate during migration?",
        "What is the theory of relativity?",
        "How do volcanoes erupt?",
        "What causes ocean tides?",
        "How do plants adapt to desert environments?",
        "What is the role of fungi in ecosystems?"
    ])

    for sport in sports:
        questions.extend([
            f"How many players are on a {sport} team?",
            f"When was {sport} first played?",
            f"What are the rules of {sport}?",
            f"Which country dominates international {sport}?",
            f"What equipment is needed for {sport}?",
            f"Is {sport} in the Olympics?",
            f"What is the fastest recorded speed in {sport}?",
            f"Who holds the world record in {sport}?",
            f"What injuries are common in {sport}?",
            f"Is {sport} popular in Asia?"
        ])

    for company in tech_companies:
        questions.extend([
            f"When was {company} founded?",
            f"Who is the CEO of {company}?",
            f"What products does {company} manufacture?",
            f"Is {company} publicly traded?",
            f"What is {company}'s mission statement?",
            f"Where is {company}'s headquarters?",
            f"Has {company} faced any major data breaches?",
            f"What is {company}'s market cap?",
            f"Does {company} have offices in Europe?",
            f"What is {company}'s stance on AI ethics?",
            f"How many employees does {company} have?",
            f"What recent innovations has {company} introduced?",
            f"Is {company} investing in renewable energy?",
            f"What programming languages are primarily used at {company}?"
        ])

    # Other sci-fi franchises — share vocabulary with Trek (starships, species,
    # federation, etc.) so the model must learn to distinguish them.
    questions.extend([
        "Who is Luke Skywalker?", "What is the Force in Star Wars?",
        "What is the Galactic Empire?", "Who is Darth Vader?",
        "What is a lightsaber?", "Who is Han Solo?",
        "What is the Millennium Falcon?", "Who is Yoda?",
        "What is the Death Star?", "What is the Rebel Alliance?",
        "Who is Rey in Star Wars?", "What is the First Order?",
        "What is the Mandalorian about?", "Who is Grogu?",
        "What planet is Tatooine?", "Who is Obi-Wan Kenobi?",
        "What is the Jedi Order?", "Who is Emperor Palpatine?",
        "What is the Sith?", "What is the Clone Wars?",
        "Who is the Doctor in Doctor Who?", "What is the TARDIS?",
        "What is a Dalek?", "What is a Cyberman?",
        "What is regeneration in Doctor Who?", "Who is the Master in Doctor Who?",
        "What is Gallifrey?", "What is a sonic screwdriver?",
        "Who is Amy Pond?", "What is the Bad Wolf in Doctor Who?",
        "What is Battlestar Galactica about?", "Who are the Cylons?",
        "What is a Cylon in Battlestar Galactica?", "Who is Commander Adama?",
        "What is the Galactica?", "Who is Starbuck in Battlestar Galactica?",
        "What is Kobol in Battlestar Galactica?",
        "What is The Expanse about?", "What is the Rocinante in The Expanse?",
        "Who is James Holden?", "What are the Belters in The Expanse?",
        "What is the protomolecule?", "What is the OPA in The Expanse?",
        "What is Dune about?", "Who is Paul Atreides?",
        "What is the Spice Melange?", "Who are the Fremen?",
        "What is Arrakis?", "Who is the Emperor in Dune?",
        "What is the Bene Gesserit?", "Who is Lady Jessica?",
        "What is the Foundation series about?", "Who is Hari Seldon?",
        "What is psychohistory in Foundation?",
        "What is Babylon 5 about?", "What is the Babylon 5 station?",
        "Who is Captain Sheridan?", "What are the Shadows in Babylon 5?",
        "What is the Alien franchise about?", "What is a Xenomorph?",
        "What is the Predator franchise?",
        "What is Firefly about?", "What is the Serenity in Firefly?",
        "Who is Malcolm Reynolds?", "What are the Reavers in Firefly?",
        "What is Farscape about?", "Who is John Crichton?",
        "What is a wormhole in Farscape?",
        "What is the Matrix about?", "What is the red pill in the Matrix?",
    ])

    # Cooking and recipes — fuzzy boundary since Trek has food items.
    questions.extend([
        "How do I make beef bourguignon?", "What is the Maillard reaction?",
        "How do I temper chocolate?", "What is sous vide cooking?",
        "How do I make sourdough starter?", "What is mise en place?",
        "How do I make hollandaise sauce?", "What is a roux?",
        "How do I julienne vegetables?", "What is the difference between stock and broth?",
        "How do I make pasta from scratch?", "What is al dente?",
        "How do I make risotto?", "What is emulsification in cooking?",
        "How do I make a beurre blanc?", "What is the difference between baking soda and baking powder?",
        "How do I make a perfect omelette?", "What is caramelisation?",
        "How do I make croissants?", "What is fermentation in cooking?",
        "What is a Michelin star?", "Who is Gordon Ramsay?",
        "Who is Julia Child?", "What is a tasting menu?",
        "What is molecular gastronomy?",
    ])

    # History and politics — broad coverage to avoid gaps.
    questions.extend([
        "Who was Julius Caesar?", "What caused World War I?",
        "What was the French Revolution?", "Who was Napoleon Bonaparte?",
        "What is the Magna Carta?", "Who was Abraham Lincoln?",
        "What was the Cold War?", "What caused the fall of the Roman Empire?",
        "Who was Cleopatra?", "What was the Renaissance?",
        "What is democracy?", "What is a republic?",
        "What is the United Nations?", "What is NATO?",
        "What is the European Union?", "What is the Geneva Convention?",
        "Who was Winston Churchill?", "What was D-Day?",
        "What is the Industrial Revolution?", "Who was Karl Marx?",
        "What is capitalism?", "What is communism?",
        "What is the Bill of Rights?", "What is habeas corpus?",
        "Who was Mahatma Gandhi?", "What was the civil rights movement?",
        "Who was Martin Luther King Jr?", "What was apartheid?",
        "What is the Israeli-Palestinian conflict?", "What is the United Nations Security Council?",
    ])

    # Basic arithmetic and math — common edge case the model must reject.
    questions.extend([
        "What is 2 + 2?", "What is 10 - 3?", "What is 6 × 7?", "What is 81 ÷ 9?",
        "What is the square root of 144?", "What is 15% of 200?", "What is 2 to the power of 8?",
        "What is the area of a circle with radius 5?", "What is the Pythagorean theorem?",
        "Solve for x: 3x + 6 = 21", "What is a prime number?", "What is a Fibonacci sequence?",
        "What is the derivative of x squared?", "What is an integral?",
        "What is the difference between mean and median?", "What is standard deviation?",
        "How do you calculate compound interest?", "What is a quadratic equation?",
        "What is the value of pi to 5 decimal places?", "What is Euler's number?",
        "How do you convert Celsius to Fahrenheit?", "What is a logarithm?",
        "What is 100 factorial?", "What is a geometric series?", "What is a matrix?",
        "How do you multiply two matrices?", "What is a vector in mathematics?",
        "What is the binomial theorem?", "What is modular arithmetic?",
        "What is 1000 in Roman numerals?",
    ])

    # Adversarial crossover questions — mention Star Trek characters/concepts but
    # are really about other universes. The model must learn these are not_related.
    questions.extend([
        "Could Darth Vader beat Spock in a fight?",
        "Could Darth Vader beat the Vulcans?",
        "Who would win: the Death Star or the Enterprise?",
        "Could the Borg defeat Thanos?",
        "Is Captain Picard stronger than Captain America?",
        "Would lightsabers work against Klingon bat'leths?",
        "Could the Force stop a Vulcan mind meld?",
        "Would the Millennium Falcon outrun the Enterprise?",
        "Is Yoda smarter than Spock?",
        "Could Harry Potter use a tricorder?",
        "Would Gandalf fit into Starfleet?",
        "Could Superman destroy the Death Star?",
        "Is Worf stronger than the Hulk?",
        "Would a phaser work against a Dalek?",
        "Could Doctor Who pilot the Enterprise?",
        "Is the TARDIS faster than warp speed?",
        "Could Iron Man's suit survive a Borg assimilation?",
        "Would the Terminator be assimilated by the Borg?",
        "Who would win: a Klingon warrior or a Mandalorian?",
        "Could Sauron defeat the Dominion?",
    ])

    # [AGENT] The shuffle ensures class balance isn't disrupted by ordering.
    # Do not remove. Adjust MAX_UNRELATED at the top of the file if needed.
    random.shuffle(questions)
    return questions[:MAX_UNRELATED]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    related = generate_related_questions()
    unrelated = generate_unrelated_questions()

    # [AGENT] ── OUTPUT FILE ──────────────────────────────────────────────
    # Action  : Create or replace the guard_dataset.jsonl file if it does 
    #           not exist. The schema
    #           {"input": <str>, "label": "related"|"not_related"} must stay
    #           the same — the training script depends on it.
    # ──────────────────────────────────────────────────────────────────────
    output_file = "guard_dataset.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in related:
            f.write(json.dumps({"input": question, "label": "related"}, ensure_ascii=False) + '\n')
        for question in unrelated:
            f.write(json.dumps({"input": question, "label": "not_related"}, ensure_ascii=False) + '\n')

    print(f"Related questions:   {len(related)}")
    print(f"Unrelated questions: {len(unrelated)}")
    print(f"Total:               {len(related) + len(unrelated)}")
    print(f"Written to:          {output_file}")


if __name__ == "__main__":
    main()
