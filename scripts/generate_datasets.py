"""Generate clearly labeled synthetic examples in separate domain folders."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ["emails", "photos", "files", "downloads", "subscriptions"]
LABELS = ["KEEP", "DELETE", "ARCHIVE", "REVIEW", "UNSUBSCRIBE"]
DOMAIN_ACTIONS = {
    "emails": LABELS,
    "photos": ["KEEP", "DELETE", "ARCHIVE", "REVIEW"],
    "files": ["KEEP", "DELETE", "ARCHIVE", "REVIEW"],
    "downloads": ["KEEP", "DELETE", "ARCHIVE", "REVIEW"],
    "subscriptions": ["KEEP", "DELETE", "ARCHIVE", "REVIEW", "UNSUBSCRIBE"],
}
TEMPLATES = {
    "emails": {
        "KEEP": ("A personal email from {sender}, subject '{subject}', received {age} days ago and opened {opens} times. Message details: {detail}. It has ongoing personal value and no sensitive attachment.", "Personal communication with ongoing value; keep it."),
        "DELETE": ("An unsolicited promotional email from {sender}, subject '{subject}', received {age} days ago and never opened. Message details: {detail}. No order, billing, or account record is present.", "Unopened stale marketing with no record-keeping value can be deleted."),
        "ARCHIVE": ("A completed transaction email from {sender}, subject '{subject}', received {age} days ago and opened {opens} times. Message details: {detail}. It contains a receipt or confirmation worth retaining.", "Keep the transaction record available by archiving it."),
        "REVIEW": ("An email from {sender}, subject '{subject}', received {age} days ago. Message details: {detail}. The sender or context is unclear, and it may involve a work or personal commitment.", "The context is insufficient to make a safe deletion decision; review it."),
        "UNSUBSCRIBE": ("A recurring newsletter from {sender}, subject '{subject}', received {age} days ago and opened {opens} times. Message details: {detail}. It includes an unsubscribe option and is no longer wanted.", "The mailing appears unwanted; unsubscribe, then remove this message if desired."),
    },
    "photos": {
        "KEEP": ("Photo metadata: {kind}; captured {age} days ago; accessed {opens} times; unique image with clear faces and high resolution.", "A unique, high-quality memory should be kept."),
        "DELETE": ("Photo metadata: {kind}; captured {age} days ago; accessed {opens} times; exact duplicate or unreadable accidental capture.", "This appears redundant or unusable and has no unique content; delete after review."),
        "ARCHIVE": ("Photo metadata: {kind}; captured {age} days ago; accessed {opens} times; meaningful event image stored in an older album.", "Preserve the image in an organized archive."),
        "REVIEW": ("Photo metadata: {kind}; captured {age} days ago; low resolution and uncertain similarity to another image.", "Uncertain image value or duplication warrants review before deletion."),
        "UNSUBSCRIBE": ("Photo item metadata: {kind}; captured {age} days ago; linked to an unwanted recurring shared-album notification.", "Remove the unwanted recurring notification or sharing subscription."),
    },
    "files": {
        "KEEP": ("File: {subject}.{ext}; modified {age} days ago; accessed {opens} times; latest approved version of an active project.", "This is the current useful version; keep it."),
        "DELETE": ("File: {subject}.{ext}; modified {age} days ago; accessed {opens} times; temporary cache or confirmed duplicate.", "The file is reproducible or redundant and has no retention value; delete after review."),
        "ARCHIVE": ("File: {subject}.{ext}; modified {age} days ago; accessed {opens} times; superseded version of a completed project.", "Retain the older version for reference in an archive."),
        "REVIEW": ("File: {subject}.{ext}; modified {age} days ago; access history is incomplete and ownership or purpose is unknown.", "Purpose and ownership are unclear; review before changing it."),
        "UNSUBSCRIBE": ("File metadata: {subject}.{ext}; relates to a recurring generated report subscription no longer used.", "Review and cancel the recurring report delivery."),
    },
    "downloads": {
        "KEEP": ("Download: {subject}.{ext}; saved {age} days ago; accessed {opens} times; active document needed by the user.", "The download is still used; keep it."),
        "DELETE": ("Download: {subject}.{ext}; saved {age} days ago; accessed {opens} times; installer or temporary archive that can be downloaded again.", "This is replaceable and unused; delete after confirming it is no longer needed."),
        "ARCHIVE": ("Download: {subject}.{ext}; saved {age} days ago; accessed {opens} times; completed tax or project record.", "Preserve the record in a clearly named archive."),
        "REVIEW": ("Download: {subject}.{ext}; saved {age} days ago; contents and source are unknown.", "Unknown contents and origin make automatic cleanup unsafe; review it."),
        "UNSUBSCRIBE": ("Download entry: {subject}.{ext}; generated by a recurring report service the user no longer uses.", "Cancel the recurring report delivery after confirming account ownership."),
    },
    "subscriptions": {
        "KEEP": ("Subscription: {subject}; last used {age} days ago; usage {opens} sessions this month; active plan needed for current work.", "Recent, purposeful use supports keeping the subscription."),
        "DELETE": ("Subscription record: {subject}; no active billing; expired trial reminder from {age} days ago.", "This is an inactive reminder with no ongoing service; remove the record if unneeded."),
        "ARCHIVE": ("Subscription record: {subject}; cancelled {age} days ago; final invoices and cancellation confirmation are retained.", "Keep billing and cancellation records in an archive."),
        "REVIEW": ("Subscription: {subject}; last used {age} days ago; renewal terms and household usage are uncertain.", "Check shared use, renewal timing, and cancellation terms before deciding."),
        "UNSUBSCRIBE": ("Subscription: {subject}; last used {age} days ago; usage {opens} sessions this month; no longer needed and renewal is approaching.", "Recommend cancellation after the user confirms there are no shared users or pending obligations."),
    },
}
VALUES = {
    "emails": {
        "KEEP": {
            "senders": ["Maya Patel", "Family Group", "Riverside School", "Northside Clinic", "Jordan Lee", "Oak Street Library", "Asha Raman", "Community Garden", "Westview Sports Club", "Daniel Kim"],
            "subjects": ["Family picnic plans", "A note from the weekend", "School concert update", "Your appointment summary", "Photos from the reunion", "Book club discussion", "Checking in after the move", "Garden volunteer schedule", "Saturday match details", "Grandpa's birthday plans"],
            "details": ["A relative shares a personal update", "A friend asks to catch up over coffee", "A teacher shares a classroom milestone", "A clinic sends care instructions", "A cousin describes a family celebration", "A librarian confirms a requested book", "A friend offers support during a move", "A neighbor coordinates a garden shift", "A teammate shares the season schedule", "A family member plans a birthday visit"],
        },
        "DELETE": {
            "senders": ["Brightmail Offers", "Weekend Bargains", "TrendCart Deals", "FreshHome Promo", "Gadget Flash Sales", "Style Street Offers", "Coupon Harbor", "Daily Retail Roundup", "Seasonal Savings", "MarketLane Promotions"],
            "subjects": ["Extra savings this weekend", "Your coupon is expiring", "New arrivals on sale", "Clearance event reminder", "A deal picked for you", "Flash sale starts today", "Seasonal offers inside", "Last chance for a discount", "Shop our weekly specials", "Limited-time price drop"],
            "details": ["A generic offer advertises kitchen accessories", "A coupon promotes a one-day clothing sale", "A product roundup links to public store pages", "A seasonal ad promotes decorative items", "A discount email lists wireless accessories", "A retailer advertises clearance home goods", "A sales blast promotes beauty products", "A promotional message features travel gear", "A weekly ad lists grocery specials", "A store advertises a short-term price cut"],
        },
        "ARCHIVE": {
            "senders": ["Acme Billing", "City Utilities", "Northstar Travel", "Harbor Insurance", "Pine Bank Statements", "Metro Tax Services", "Brightline Electronics", "Willow Pharmacy", "Cedar Rentals", "Summit Learning"],
            "subjects": ["Paid invoice for March", "Electricity payment receipt", "Hotel booking confirmation", "Policy renewal receipt", "Monthly account statement", "Tax submission confirmation", "Laptop order receipt", "Prescription payment record", "Rental deposit receipt", "Course completion receipt"],
            "details": ["A paid invoice records the final amount and date", "A utility receipt confirms the balance was paid", "A hotel confirms a completed stay and booking ID", "An insurer confirms receipt of a premium payment", "A bank statement documents a completed billing period", "A tax service confirms a submitted return", "A retailer records a fulfilled equipment order", "A pharmacy confirms a completed prescription purchase", "A property manager documents a returned deposit", "A learning service records a completed course payment"],
        },
        "REVIEW": {
            "senders": ["Unknown Sender", "Workplace Systems", "Community Notes", "Alex Morgan", "Service Desk", "Events Committee", "Account Services", "Project Coordinator", "Local Association", "Benefits Office"],
            "subjects": ["Please respond when you can", "Updated project information", "Invitation details inside", "Question about your request", "Action may be needed", "Schedule change notice", "Document ready to view", "Follow-up on your account", "Meeting details changed", "Information for your records"],
            "details": ["The message omits the prior conversation", "The sender address does not match the display name", "The attachment purpose is not explained", "It is unclear whether the request is still active", "The message may relate to a personal appointment", "A link is included without context", "The thread references an unknown project", "The recipient list suggests a shared account", "The message asks for a response but gives no deadline", "The content could be a receipt or a routine notice"],
        },
        "UNSUBSCRIBE": {
            "senders": ["Daily Digest", "Brightmail Newsletter", "Tech Bulletin", "Garden Monthly", "Travel Ideas Weekly", "Food Notes", "Design Dispatch", "Campus Updates", "Bookwire Roundup", "Market Insights"],
            "subjects": ["This week's editor picks", "Monthly product newsletter", "New stories from our team", "Garden notes for April", "Ideas for your next trip", "Fresh recipes this month", "Design trends to explore", "Campus news roundup", "Books you may enjoy", "The monthly market brief"],
            "details": ["A recurring editorial email offers weekly reading links", "A mailing list sends product news each month", "A publisher shares regular articles and announcements", "A gardening group sends a scheduled monthly issue", "A travel site sends recurring destination ideas", "A cooking newsletter delivers recipes every week", "A design publication sends its regular digest", "A university list shares routine campus updates", "A bookseller sends a recurring recommendations email", "A market newsletter provides its regular commentary"],
        },
    },
    "photos": {
        "KEEP": ["unique family portrait", "one-of-a-kind travel photograph", "high-resolution celebration photo"],
        "DELETE": ["confirmed exact duplicate", "accidental pocket photo", "corrupt unreadable thumbnail"],
        "ARCHIVE": ["meaningful completed-event photograph", "older album image retained for memories", "original-resolution event photo"],
        "REVIEW": ["low-resolution image with uncertain similarity", "photo with unclear people and context", "possible duplicate without a reliable match"],
    },
    "files": {
        "KEEP": [("active-project-final", "docx"), ("current-budget", "xlsx"), ("approved-design", "pdf")],
        "DELETE": [("confirmed-duplicate-copy", "pdf"), ("reproducible-cache", "tmp"), ("temporary-installer", "zip")],
        "ARCHIVE": [("completed-project-v1", "docx"), ("prior-year-tax-record", "pdf"), ("superseded-approved-version", "xlsx")],
        "REVIEW": [("unknown-owner-document", "docx"), ("unidentified-archive", "zip"), ("unclear-purpose-record", "pdf")],
    },
    "downloads": {
        "KEEP": [("active-work-reference", "pdf"), ("current-course-material", "docx"), ("needed-project-data", "xlsx")],
        "DELETE": [("unused-replaceable-installer", "zip"), ("temporary-download-cache", "tmp"), ("confirmed-duplicate-download", "pdf")],
        "ARCHIVE": [("completed-tax-record", "pdf"), ("finished-project-deliverable", "docx"), ("past-booking-confirmation", "pdf")],
        "REVIEW": [("unknown-source-download", "zip"), ("unidentified-document", "pdf"), ("unclear-contents-archive", "zip")],
    },
    "subscriptions": {
        "KEEP": ["current design plan", "active cloud backup", "work collaboration plan"],
        "DELETE": ["expired free trial with no active billing", "cancelled service with no remaining charge", "obsolete inactive account reminder"],
        "ARCHIVE": ["cancelled plan with final invoice", "closed account with tax receipts", "ended service with cancellation confirmation"],
        "REVIEW": ["renewal terms unclear and household use unknown", "shared account usage not verified", "cancellation may affect stored data"],
        "UNSUBSCRIBE": ["unwanted newsletter subscription with unsubscribe link", "recurring promotional mailing no longer wanted", "marketing email list with unsubscribe option"],
    },
}


def create_record(domain: str, index: int, rng: random.Random) -> dict[str, str | bool]:
    action = DOMAIN_ACTIONS[domain][index % len(DOMAIN_ACTIONS[domain])]
    template, reason = TEMPLATES[domain][action]
    if domain == "emails":
        scenarios = VALUES["emails"][action]
        action_slot = index // len(DOMAIN_ACTIONS[domain])
        sender = scenarios["senders"][action_slot % len(scenarios["senders"])]
        subject = scenarios["subjects"][(action_slot // len(scenarios["senders"])) % len(scenarios["subjects"])]
        detail = scenarios["details"][action_slot % len(scenarios["details"])]
        # Engagement and age reinforce, rather than contradict, the label.
        age = rng.choice([45, 120, 365, 900]) if action == "DELETE" else rng.choice([1, 2, 14, 45])
        opens = 0 if action == "DELETE" else rng.choice([1, 2, 8])
        text = template.format(sender=sender, subject=subject, detail=detail, age=age, opens=opens)
    elif domain == "photos":
        value = rng.choice(VALUES[domain][action])
        text = template.format(kind=value, age=rng.choice([5, 30, 180, 700]), opens=rng.choice([0, 1, 3]))
    elif domain in {"files", "downloads"}:
        value = rng.choice(VALUES[domain][action])
        subject, ext = value
        text = template.format(subject=subject, ext=ext, age=rng.choice([2, 30, 180, 700]), opens=rng.choice([0, 1, 3]))
    else:
        value = rng.choice(VALUES[domain][action])
        text = template.format(subject=value, age=rng.choice([2, 30, 180, 700]), opens=rng.choice([0, 1, 3]))
    return {"id": f"{domain}-{index + 1:05d}", "domain": domain[:-1] if domain != "files" else "file", "item_text": text, "actual_action": action, "reason": reason, "synthetic": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-per-domain", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-root", type=Path, default=ROOT / "datasets_v2", help="Write to a separate dataset root (defaults to datasets_v2).")
    args = parser.parse_args()
    if args.records_per_domain < len(LABELS):
        parser.error("records-per-domain must be at least 5 to include every action")
    for domain_index, domain in enumerate(DOMAINS):
        folder = args.output_root / domain
        (folder / "raw").mkdir(parents=True, exist_ok=True)
        (folder / "curated").mkdir(parents=True, exist_ok=True)
        (folder / "splits").mkdir(parents=True, exist_ok=True)
        rng = random.Random(args.seed + domain_index)
        records = [create_record(domain, i, rng) for i in range(args.records_per_domain)]
        jsonl = folder / "raw" / "synthetic_records.jsonl"
        csv_path = folder / "raw" / "synthetic_records.csv"
        jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        (folder / "curated" / "README.md").write_text(
            "# Human-reviewed records\n\nPlace only human-reviewed, approved examples here before training. The generator does not populate this folder or claim synthetic labels have been reviewed. See the domain README and project README for the schema.\n",
            encoding="utf-8",
        )
        print(f"{domain}: wrote {len(records)} synthetic examples to {jsonl}")


if __name__ == "__main__":
    main()
