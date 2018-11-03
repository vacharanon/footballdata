import datetime
def next_draw_method(fixtures, starting_balance=1000):
#     starting_balance = 1000
    balance = starting_balance
    bet = -50
    balances = [1000]
    dates = [datetime.datetime(fixtures['Date'].iloc[0].year,8,1)]
    receives = [0]
    bets = [0]
    profits = [0]
    print("Starting balance:", balance)
    for date, groupByDate in fixtures.groupby('Date'):
        bet_today = 0
        receive_today = 0
        for _, match in groupByDate.iterrows():
    #         print(match['index'])
            d = date.date().strftime('%Y-%m-%d')
            predictions = total[total.index == match['index']]
            if (len(predictions) == 1):
    #             prediction = predictions[:,0]
    #             print(predictions)
                team = predictions['Team'].values[0]
                pred = predictions['Predict'].values[0]
                act = predictions['Actual'].values[0]
                balance += bet
                bet_today += bet
                print(d, "Pay:", bet, 'on', team, 'in [', match['HomeTeam'], 'vs', match['AwayTeam'], "] Pred:", pred, "Actual:", act, 'Balance:', balance)
                
                odd = 0
                if pred == act:
    #                 print(match)
                    if pred == act == 'Draw':
                        odd = match['B365D']
                    elif match['HomeTeam'] == team:
                        odd = match['B365H']
                    else:
                        odd = match['B365A']
                    receive = bet * odd * -1
                    balance += receive
                    receive_today += receive
                    print(d, "Receive:", receive, "Odd:", odd, "Balance:", balance)
            elif (len(predictions) == 2):
    #             print(predictions)
    #             print(match)
                pred = ''
                act = ''
                pred_one = predictions['Predict'].values[0]
                pred_two = predictions['Predict'].values[1]
                act_one = predictions['Actual'].values[0]
                act_two = predictions['Actual'].values[1]
    #             print(team, "Pred: ", pred, "Actual:", act)
                odd = 0
                # unanimousness
                if (pred_one == pred_two == 'Draw') or\
                (pred_one == 'Win' and pred_two == 'Lose') or\
                (pred_one == 'Lose' and pred_two == 'Win'):

#                     balance += bet
#                     print('what', len(predictions), balance)
#                     print(match)
#                     odd = 0

                    if pred_one == pred_two == act_one == act_two == 'Draw':
                        team = predictions['Team'].values[0]
                        odd = match['B365D']
                        pred = pred_one
                        act = act_one
#                         print(d, "Pay:", bet, 'on', team, 'in [', match['HomeTeam'], 'vs', match['AwayTeam'], "] Pred:", pred, "Actual:", act, 'Balance:', balance)

                    elif pred_one == act_one == 'Win' and pred_two == act_two == 'Lose':
                        team = predictions['Team'].values[0]
                        if match['HomeTeam'] == team:
                            odd = match['B365H']
                        else:
                            odd = match['B365A']
                        pred = pred_one
                        act = act_one
#                         print(d, "Pay:", bet, 'on', team, 'in [', match['HomeTeam'], 'vs', match['AwayTeam'], "] Pred:", pred_one, "Actual:", act_one, 'Balance:', balance)

                    elif pred_one == act_one == 'Lose' and pred_two == act_two == 'Win':
                        team = predictions['Team'].values[1]
                        if match['HomeTeam'] == team:
                            odd = match['B365H']
                        else:
                            odd = match['B365A']
                        pred = pred_two
                        act = act_two
#                         print(d, "Pay:", bet, 'on', team, 'in [', match['HomeTeam'], 'vs', match['AwayTeam'], "] Pred:", pred_two, "Actual:", act_two, 'Balance:', balance)
                    else:
                        team = predictions['Team'].values[0]
#                         print(d, "Pay:", bet, 'on', team, 'in [', match['HomeTeam'], 'vs', match['AwayTeam'], "] Pred:", pred_one, "Actual:", act_one, 'Balance:', balance)
                elif pred_one == 'Draw' or pred_two == 'Draw':
                    # if one of them predict Draw but the other does not, find next best
                    if pred_one == 'Draw':
                        lose_prob_two = predictions['LoseProb'].values[1]
                        win_prob_two = predictions['WinProb'].values[1]
                        if lose_prob_two > win_prob_two:
                            # guess a draw
                            team = predictions['Team'].values[0]
                            pred = 'Draw'
                            if act_one == 'Draw':
                                odd = match['B365D']
#                         else:
#                             # do nothing
#                             pass
                    else:
                        lose_prob_one = predictions['LoseProb'].values[0]
                        win_prob_one = predictions['WinProb'].values[0]
                        if lose_prob_one > win_prob_one:
                            # guess a draw
                            team = predictions['Team'].values[1]
                            pred = 'Draw'
                            if act_one == 'Draw':
                                odd = match['B365D']
                elif pred_one == pred_two == 'Lose':
                    # if both are losers, bet on draw
                    team = predictions['Team'].values[0]
                    pred = 'Draw'
                    if act_one == 'Draw':
                        odd = match['B365D']
                    act = act_one
                if pred != '':
                    balance += bet
                    bet_today += bet
                    print(d, "Pay:", bet, 'on', team, 'in [', match['HomeTeam'], 'vs', match['AwayTeam'], "] Pred:", pred, "Actual:", act, 'Balance:', balance)
                else:
                    print(d, "No bet in [", match['HomeTeam'], 'vs', match['AwayTeam'], "] Balance:", balance)
                    
                if odd != 0:
                    receive = bet * odd * -1
                    balance += receive
                    receive_today += receive
                    print(d, "Receive:", receive, "Odd:", odd, "Balance:", balance)
                

            else:
                print(d, "No bet in [", match['HomeTeam'], 'vs', match['AwayTeam'], "] Balance:", balance)
    #             print(predictions)
        balances.append(balance)
        dates.append(date)
        receives.append(receive_today)
        bets.append(bet_today)
        profits.append(receive_today + bet_today)
    print("Ending balance:", balance)
    year = dates[0].date().year
    df_sim = pd.DataFrame(
    data={
        'Date': dates,
        'Balance': balances,
        'Year': year,
        'MatchDay': range(1, len(dates) + 1),
        'Receive': receives,
        'Bet': bets,
        'Profit': profits
    })
    return df_sim