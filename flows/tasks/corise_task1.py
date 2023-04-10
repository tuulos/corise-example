import io
import pandas
import random

from metaflow import FlowSpec, step, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact

class Task1_Clothing(FlowSpec):
	
	data = IncludeFile('data', default='Womens Clothing E-Commerce Reviews.csv')

	@step
	def start(self):
		self.df = pandas.read_csv(io.StringIO(self.data))
		self.next(self.train)

	@step
	def train(self):
		self.df['predictions'] = [random.randint(0, 1) for _ in range(len(self.df))]
		self.next(self.end)

	@card(type='corise')
	@step
	def end(self):
		correct = len(self.df[self.df['predictions'] == self.df['Recommended IND']]) / len(self.df)
		current.card.append(Markdown("# Womens Clothing Review Results"))

		current.card.append(Markdown("## Overall Accuracy"))
		current.card.append(Artifact(correct))		

		current.card.append(Markdown("## Accuracy by Department"))
		grouped = self.df.groupby(['Department Name']).apply(lambda g: g[g['predictions'] == g['Recommended IND']].count() / g.count())
		current.card.append(Table.from_dataframe(grouped[['predictions']]))

		current.card.append(Markdown("## Examples of False Positives"))
		false_pos = self.df[(self.df['predictions'] == 1) & (self.df['Recommended IND'] == 0)][:10]
		#print('false po', len(false_pos))
		current.card.append(Table.from_dataframe(false_pos[['Review Text']]))

		current.card.append(Markdown("## Examples of False Negatives"))
		false_neg = self.df[(self.df['predictions'] == 0) & (self.df['Recommended IND'] == 1)][:10]
		current.card.append(Table.from_dataframe(false_neg[['Review Text']]))


if __name__ == '__main__':
	Task1_Clothing()