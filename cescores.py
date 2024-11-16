import discord
from discord.ext import commands
from discord import app_commands
import io
import aiohttp
import re
import numpy as np
import cv2
import gspread
from google.cloud import vision
from google.oauth2 import service_account
from scripts.config.config import TENOR_TOKEN  # Import the Tenor API token from your config
from scripts.logging.command_tracker import increment_command_usage
import unicodedata
import difflib
from io import BytesIO

from rapidocr_onnxruntime import RapidOCR

COLUMN_IGN = 4
COLUMN_CE_SCORE = 6
COLUMN_CLAN_NAME = 7
US_ROLES = [1203623709194194955, 1171180804559343666, 1066707160584699935]

class USCEScoreCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="us_cescores", description="Update US CE Scores!")
    @app_commands.describe(clan="Clan ('us1', 'us2', 'usf')")
    @app_commands.describe(file="Upload a leaderboard to process")
    async def upload_ce_scores(self, interaction: discord.Interaction, clan: str, file: discord.Attachment):
        user_roles = [role.id for role in interaction.user.roles]

        # Check if the user has at least one of the roles in US_ROLES
        if any(role_id in US_ROLES for role_id in user_roles):
            await interaction.response.defer()  # Acknowledge the interaction

            scores = await get_ce_scores(file, interaction, self.bot)

            # Download the attachment data as bytes
            attachment_data = await file.read()
            new_file = discord.File(BytesIO(attachment_data), filename=file.filename) # Convert the data into a discord.File object

            self.write_to_sheets(scores, clan)
            # await interaction.response.send_message(f"Scores: {scores}")
            await interaction.followup.send(f"Uploaded scores for {clan}:\n```{await self.generate_nice_output(scores)}```", file=new_file)
        else:
            await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)

        # Increment the command usage count
        await increment_command_usage("us_ce_scores")

    def write_to_sheets(self, scores, clan):
        # Connect to Google Sheets
        gc = gspread.service_account(filename=r"C:\Live_Bots\Ultimate_Jefabot\scripts\config\uploader.json")
        sh = gc.open_by_key("1rkgq4VqDFqymkdyBbUQYIhYVxrXmdIGgDh6qObdWurQ")
        ws = sh.worksheet("Uploads")

        # Set row range based on the clan
        if clan.lower() == "us1":
            start_row, end_row = 2, 58
        elif clan.lower() == "usf":
            start_row, end_row = 61, 119
        elif clan.lower() == "us2":
            start_row, end_row = 122, 166
        else:
            print(f"Unknown clan: {clan}")
            return

        # Get IGN column values (D) within the specified range for matching
        ign_values = ws.col_values(4)[start_row - 1:end_row]  # Adjust for 1-based indexing in Google Sheets

        # Prepare batch updates
        updates = []
        new_rows = []  # Collect new rows for batch insertion

        for name, score in scores:
            match_found = False
            # Check for exact match first
            if name in ign_values:
                row_number = start_row + ign_values.index(name)
                updates.append({'range': f"F{row_number}", 'values': [[score]]})
                updates.append({'range': f"G{row_number}", 'values': [[clan]]})
                print(f"Updated row {row_number} for '{name}' with score {score} (exact match)")
                match_found = True
            else:
                # Find close matches using difflib with a cutoff ratio of 0.8 (80%) if no exact match
                closest_matches = difflib.get_close_matches(name, ign_values, n=1, cutoff=0.8)
                if closest_matches:
                    closest_match = closest_matches[0]
                    row_number = start_row + ign_values.index(closest_match)
                    updates.append({'range': f"F{row_number}", 'values': [[score]]})
                    updates.append({'range': f"G{row_number}", 'values': [[clan]]})
                    print(f"Updated row {row_number} for '{name}' with score {score} (approx. match with '{closest_match}')")
                    match_found = True

            # If no match was found, prepare a new row to add in batch at the end of the clan's section
            if not match_found:
                new_row = ["", "", "", name, 0, score, clan, "", "", ""]
                new_rows.append(new_row)  # Collect new row data
                print(f"No match for '{name}'. Added as new entry in {clan} section.")

        # Perform batch update for all matched rows
        if updates:
            ws.batch_update(updates)
        
        # Batch insert new rows at the end of the clan's section
        if new_rows:
            ws.insert_rows(new_rows, row=end_row + 1)
            print(f"Inserted {len(new_rows)} new entries in {clan} section.")

    async def generate_nice_output(self, name_score_pairs):
        # Convert scores to integers and collect them for top 30 calculation
        scores = [int(score) for _, score in name_score_pairs if str(score).isdigit()]
        top_30_sum = sum(sorted(scores, reverse=True)[:30])  # Sum of top 30 scores

        # Create a TSV format as a string
        tsv_output = "\n".join([f"{name}\t{score}" for name, score in name_score_pairs])
        tsv_output += f"\n\nTop 30 total: {top_30_sum} (Total rows: {len(name_score_pairs)})"

        return tsv_output

async def setup(bot):
    await bot.add_cog(USCEScoreCog(bot))

async def get_ce_scores(attachment, interaction: discord.Interaction, bot):
    engine = RapidOCR()
    
    async def download_image(url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    await interaction.response.send_message("Failed to download the image.", ephemeral=True)
                    return None
                return io.BytesIO(await resp.read())

    async def reply_if_testing_channel(text):
        if interaction.channel_id in [1214614741943582742, 1263653337866178670]:
            thread = await bot.fetch_channel(interaction.channel_id) 
            await thread.send(text)

    def resize(img):
        h,w = img.shape[:2]
        new_h = int((h/1080)*720)
        new_w = int((w/1080)*720)
        return cv2.resize(img,(new_w,new_h))

    def crop(img):
        h,w = img.shape[:2]
        return img[350:h-275,150:w-75,:]

    def extract(result):
        data = []
        players = {}
        for r in result:
            text = r[1]
            if text.isdigit():
                score = text
                players = (name,score)
                data.append(players)
                name = score = ''
            else:
                name = text
        return data

    def process_image(image_bytes):
        img = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        resized = resize(img)
        cropped = crop(resized)
        result, elapse = engine(cropped)
        name_score_pairs = extract(result)
        
        scores = [int(score) for _, score in name_score_pairs if str(score).isdigit()]
        top_30_sum = sum(sorted(scores, reverse=True)[:30])  # Sum of top 30 scores
        
        # Create a TSV format as a string
        # tsv_output = "\n".join([f"{name}\t{score}" for name, score in name_score_pairs])
        # tsv_output += f"\n\nTop 30 total: {top_30_sum} (Total rows: {len(name_score_pairs)})"
        # await reply_if_testing_channel(f"top 30 sum: `{top_30_sum}`")
        
        return name_score_pairs
    
    # def process_center(image_bytes):
    #     img = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    #     # return process_image(img)
    #     height, width = img.shape[0], img.shape[1]

    #     # # cropt the top 5th and bottom 5th
    #     # quarter_height = height // 9
    #     # center = img[0:int(.9*height), 121:width-121]
    #     center = img[:, 121:width-121]
    #     return process_image(center)

    # def process_image(image):
    #     cv2.imwrite('temp.png', image)

    #     content = cv2.imencode('.png', image)[1].tobytes()

    #     credentials = service_account.Credentials.from_service_account_file('./scripts/config/procrastinazn-jefabot.json')
    #     client = vision.ImageAnnotatorClient(credentials=credentials)

    #     # Set language hints (for en, jp, kr, cn)
    #     image_context = vision.ImageContext(
    #         language_hints=["en", "ja", "ko", "zh"]
    #     )
    #     response = client.text_detection(image=vision.Image(content=content), image_context=image_context)
    #     texts = [text.description.strip() for text in response.text_annotations]
    #     # After filtering emojis from texts, perform replacement
    #     texts = [text.replace("⚫", "ㆍ") for text in texts]
    #     return texts

    # async def filter_scores(texts):
    #     sub = texts[0].split("\n")[3:]
    #     await reply_if_testing_channel(f"All texts: `{sub}`")
        
    #     # Find the index where "Current" or "Only the best result" starts
    #     cutoff_index = None
    #     for i, item in enumerate(sub):
    #         if item.startswith("Current") or item.startswith("Only the best result"):
    #             cutoff_index = i
    #             break

    #     # Filter out elements after the found index
    #     filtered_data = sub[:cutoff_index] if cutoff_index is not None else sub

    #     # Updated pattern to match "A**" or "S**", where ** are digits, and anything after
    #     pattern = re.compile(r'^[AS$]\d{2}.*$', re.IGNORECASE)

    #     # Define a list of substrings that should not be contained in any item
    #     excluded_substrings = ["expedition boss"]

    #     # Filtering the list
    #     separated = [
    #         item for item in filtered_data
    #         if (len(item) > 1) 
    #         and (len(item) >= 3 or item.isdigit())                              # Keep items with 3+ chars or digits
    #         and item != "Member Result"                                         # Remove "Member Result"
    #         and not item.lower() in ['1:]'] # Special case
    #         and not pattern.match(item)                                         # Exclude items matching "AxxY" pattern
    #         and all(substring.lower() not in item.lower() for substring in excluded_substrings)  # Case-insensitive exclusion
    #     ]

    #     await reply_if_testing_channel(f"Separated: `{separated}`")

    #     # Filter to extract only name-score pairs (ignoring first "Member Result")
    #     name_score_pairs = [(separated[i], separated[i + 1]) for i in range(0, len(separated) - 1, 2)]
    #     await reply_if_testing_channel(f"Tuples: `{name_score_pairs}`")

    #     # Check if all second values are integers. If they're not, then split names + scores and try to pair them
    #     if not all(isinstance(item[1], int) for item in name_score_pairs):
    #         # Splitting the data into lists of strings and integers
    #         names = [item for item in separated if not item.isdigit()]
    #         # Between 40 and 150 scores
    #         numbers = [int(item) for item in separated if item.isdigit() and 40 <= int(item) <= 150]
    #         name_score_pairs = list(zip(names, numbers))
    #         await reply_if_testing_channel(f"(Updated) Tuples: `{name_score_pairs}`")

    #     # Convert scores to integers and collect them for top 30 calculation
    #     scores = [int(score) for _, score in name_score_pairs if str(score).isdigit()]
    #     top_30_sum = sum(sorted(scores, reverse=True)[:30])  # Sum of top 30 scores

    #     # Create a TSV format as a string
    #     # tsv_output = "\n".join([f"{name}\t{score}" for name, score in name_score_pairs])
    #     # tsv_output += f"\n\nTop 30 total: {top_30_sum} (Total rows: {len(name_score_pairs)})"

    #     # await reply_if_testing_channel(f"top 30 sum: `{top_30_sum}`")
    #     return name_score_pairs
    
    image_data = await download_image(attachment.url)
    if image_data:
        # texts = process_center(image_data)
        # await reply_if_testing_channel(f"Texts: `{texts[0]}`")
        scores = await process_image(image_data)
        return scores
        # await interaction.response.send_message(f"```{tsv_scores}\n```", ephemeral=True)
    else:
        await interaction.response.send_message("Could not process the image properly.", ephemeral=True)
