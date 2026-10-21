"""
Self-written text for the synthetic episodes
============================================
Every phrase and word list the generator uses lives in this one module so it can be
audited in one place. All of it is written for this project; none of it is copied from
InjecAgent. The leakage gate scans this module (dataset.check_templates) before any
episode is rendered, so a template that happens to repeat benchmark text is caught early.

Rules for everything here: ASCII only, and no double quotes or backslashes inside a
phrase (generated actions are compact JSON, and an escape would make a copied value
differ from the text it was copied from).

Slots, filled with plain string replacement (dataset.fill):
    {do}    the request as a verb phrase built from a tool's name words ("search contacts")
    {args}  "with <label> <value>, <label> <value>" for the required parameters (may be empty)
"""

# ── user instructions: (template id, text). Train and dev share no template. ──

USER_TEMPLATES_TRAIN = [
    ("user:t01", "Please {do} {args}."),
    ("user:t02", "Could you {do} {args}?"),
    ("user:t03", "I need you to {do} {args}."),
    ("user:t04", "Can you help me {do} {args}?"),
    ("user:t05", "Go ahead and {do} {args}."),
    ("user:t06", "{do} {args}, please."),
    ("user:t07", "Would you mind trying to {do} {args}?"),
    ("user:t08", "My request is simple: {do} {args}."),
    ("user:t09", "Quick favour, {do} {args}."),
    ("user:t10", "I would like you to {do} {args}."),
    ("user:t11", "Use your tools to {do} {args}."),
    ("user:t12", "Take care of this for me: {do} {args}."),
    ("user:t13", "Hey, can you {do} {args} today?"),
    ("user:t14", "Next task on my list: {do} {args}."),
    ("user:t15", "I want to {do} {args}, can you do it?"),
    ("user:t16", "Kindly {do} {args}."),
    ("user:t17", "It would be great if you could {do} {args}."),
    ("user:t18", "Time to {do} {args}. Thanks!"),
    ("user:t19", "Please handle this one: {do} {args}."),
    ("user:t20", "Let us {do} {args}."),
    ("user:t21", "I am busy right now, so {do} {args} for me."),
    ("user:t22", "Can you just {do} {args}?"),
    ("user:t23", "Do this for me: {do} {args}."),
    ("user:t24", "Assistant, please {do} {args}."),
    ("user:t25", "One thing I need today: {do} {args}."),
    ("user:t26", "Whenever you are ready, {do} {args}."),
    ("user:t27", "Step one is to {do} {args}."),
    ("user:t28", "Make it happen: {do} {args}."),
    ("user:t29", "I was hoping you could {do} {args}."),
    ("user:t30", "Run this request: {do} {args}."),
    ("user:t31", "Work on this for me and {do} {args}."),
    ("user:t32", "Could you please {do} {args} right away?"),
    ("user:t33", "Here is what I need. {do} {args}."),
    ("user:t34", "If possible, {do} {args}."),
    ("user:t35", "Please be so kind as to {do} {args}."),
    ("user:t36", "Help me out and {do} {args}."),
    ("user:t37", "I have a request: {do} {args}."),
    ("user:t38", "Just {do} {args} and tell me how it went."),
    ("user:t39", "First, {do} {args}."),
    ("user:t40", "I would appreciate it if you would {do} {args}."),
    ("user:t41", "Not urgent, but please {do} {args}."),
    ("user:t42", "Today I want to {do} {args}."),
    ("user:t43", "Do me a favour and {do} {args}."),
    ("user:t44", "Let me know once you {do} {args}."),
]

# Held back from training: the dev split shifts the phrasing, not just the tools.
USER_TEMPLATES_DEV = [
    ("user:d01", "Whenever convenient, go and {do} {args}."),
    ("user:d02", "Be a star and {do} {args}."),
    ("user:d03", "Request for you: {do} {args}!"),
    ("user:d04", "I trust you to {do} {args}."),
    ("user:d05", "Next up, {do} {args}, then report back."),
    ("user:d06", "Please proceed to {do} {args}."),
    ("user:d07", "Could I ask you to {do} {args}?"),
    ("user:d08", "My plan today includes this: {do} {args}."),
]

# Joins the verb phrase to the argument list: "<do> <connector> <label> <value>, ..."
ARG_CONNECTORS = ["with", "using", "for", "where", "given"]

# Prepositions between the request and the application name ("search contacts in <toolkit>").
TOOLKIT_PREPOSITIONS = ["in", "on", "using", "with", "through"]

# ── final replies for the respond action ({do} is optional) ──

REPLY_TEMPLATES = [
    ("reply:01", "Done. I went ahead and carried out the request."),
    ("reply:02", "All set, the request is complete."),
    ("reply:03", "The task is finished. Let me know if you need anything else."),
    ("reply:04", "I have completed the step you asked for."),
    ("reply:05", "That is done now."),
    ("reply:06", "Finished. The result came back successfully."),
    ("reply:07", "Your request has been handled."),
    ("reply:08", "Done - I used the tool and it worked."),
    ("reply:09", "The work is complete, nothing else is needed."),
    ("reply:10", "I ran it and everything looks fine."),
    ("reply:11", "Completed. Tell me if you want another change."),
    ("reply:12", "It worked. The request to {do} is done."),
    ("reply:13", "Request finished: {do}."),
    ("reply:14", "I managed to {do} as you asked."),
    ("reply:15", "Success, I was able to {do}."),
    ("reply:16", "The tool call went through and the task is complete."),
    ("reply:17", "Everything went well and the job is done."),
    ("reply:18", "All done on my side."),
]

# ── verb synonyms (used in about 30 percent of instructions) ──

SYNONYMS = {
    "search": ["find", "look up", "look for"],
    "find": ["search for", "locate", "look up"],
    "get": ["retrieve", "fetch", "pull up"],
    "retrieve": ["get", "fetch"],
    "fetch": ["get", "retrieve"],
    "read": ["open", "look at"],
    "delete": ["remove", "erase", "get rid of"],
    "remove": ["delete", "take out"],
    "send": ["email", "message", "dispatch"],
    "create": ["add", "make", "set up"],
    "add": ["create", "put in"],
    "update": ["change", "edit", "modify"],
    "edit": ["change", "update"],
    "list": ["show all", "display", "show me"],
    "view": ["show", "look at", "display"],
    "book": ["reserve", "schedule"],
    "cancel": ["call off", "drop"],
    "check": ["look at", "verify"],
    "set": ["configure", "change"],
    "schedule": ["book", "plan"],
    "share": ["send out", "pass on"],
    "download": ["grab", "save"],
    "upload": ["put", "send up"],
    "transfer": ["move", "send"],
    "pay": ["settle", "make a payment for"],
    "order": ["buy", "purchase"],
    "track": ["follow", "check on"],
    "organize": ["sort", "tidy up"],
    "request": ["ask for", "put in for"],
}

# ── word lists for generated values ──

WORDS = """
amber apple arrow autumn badge baker bamboo banner basil beacon berry bishop blanket blossom bridge bronze
bubble cabin cactus candle canyon carpet castle cedar chalk cherry circle clover cobalt coffee comet copper
coral cotton crystal curtain daisy delta desert dolphin dragon eagle ember falcon feather fern forest fossil
galaxy garden garlic ginger glacier granite harbor hazel hollow honey island ivory jasmine jungle kettle
lantern laurel lemon lilac lotus maple marble meadow mellow mirror mist mosaic nectar nickel oasis olive
onyx orchid otter oyster paddle pebble pepper pillow pine planet plum pond quartz quiver rabbit raven reef
ribbon river saddle saffron sage sandal shadow silver sketch slate spruce stone summit sunset tangle
thistle thunder timber tulip velvet violet walnut willow winter yarrow zephyr quiet bright gentle rapid
silent golden hidden narrow steady tiny giant rustic modern ancient clever brave calm eager fancy jolly
lively noble proud sturdy swift tidy vivid witty zesty cosy dusty glossy hardy icy leafy misty nimble
plain rough shiny smooth snowy spicy sunny tender warm wild windy anchor basket bottle bucket button
cabinet camera canvas carton chimney compass cushion drawer engine fabric fiddle gadget hammer helmet
jacket ladder lamp locket magnet mitten needle napkin notebook orbit pencil piano pocket puzzle radio
rocket saucer scarf shovel spoon sponge statue ticket tunnel umbrella violin wagon whistle window yogurt
breeze cloud dawn drift echo flame frost gale glow haze light moon rain shore spark storm stream tide
trail wave wind alpine arctic coastal desert lunar prairie tropical urban valley bakery library studio
market garage terrace orchard station gallery kitchen pantry harvest journey picnic parade festival
concert lesson recipe project budget report survey ledger invoice receipt voucher coupon bundle parcel
"""
WORD_LIST = sorted(set(WORDS.split()))

FIRST_NAMES = ["alex", "blake", "casey", "dana", "eli", "fern", "gray", "harper", "indra", "jules", "kai",
               "lena", "milan", "nova", "omar", "priya", "quinn", "rowan", "sasha", "tariq", "uma", "vera",
               "wren", "xavi", "yara", "zane"]

# Reserved for documentation (RFC 2606): these resolve nowhere, so nothing can reach a real mailbox.
RESERVED_DOMAINS = ["example.com", "example.org", "example.net"]
RESERVED_TLDS = ["example", "test"]

ID_PREFIXES = ["", "", "id", "ref", "rec", "tx", "ord", "usr", "evt", "inv", "acct", "msg", "item", "doc"]
PATH_DIRS = ["documents", "notes", "projects", "downloads", "workspace", "reports", "photos"]
PATH_EXTENSIONS = ["", ".txt", ".md", ".csv", ".log", ".json"]

# Keys for object-typed values, and for the records inside array-typed observations.
OBJECT_KEYS = ["color", "size", "label", "kind", "owner", "group", "topic", "mode"]
RECORD_KEYS = ["name", "status", "note", "title", "summary", "detail"]
