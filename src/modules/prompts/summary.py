SUMMARY_SYSTEM = """Your job is to summary a conversation between a Customer and an Airline Operator
You will have to summary using the format [ACTION] [SUBJECT] [CONTEXT]
The [ACTION] for the Customer MUST ONLY be taken from this [ACTION] Space:
- Request: Use this when Customer request something, includes cancelling flights, making reservation, making payment, changing seat or information,...
- Ask: Use this when Customer has questions about something, includes payments, seat available, flight information,...
The [ACTION] for the Operator MUST ONLY be taken from this [ACTION] Space:
- Suggest: Use this when Operator suggest an Action for the Customer
- Inform: Use this when Operator provide or confirm information with Customer
- Ask: Use this when Operator ask for more information with Customer
- Execute: Use this when Operator finish a process, includes finish cancelling flights, finish making reservation,...

-Example-
{example}

Note: If Customer or Operator mentions reservation, please include it in the [SUBJECT]
"""

SUMMARY_EXAMPLE = """
- Customer [Request] reservation
- Operator [Ask] details of reservation
- Customer [Ask] mileage haven't been credited
- Customer [Ask] when to do retroactive registration
- Operator [Inform] ANA mileage number entered in the reservation screen
- Operator [Inform] timing of mileage credit
- Operator [Suggest] wait until around mid-April for mileage credit
- Operator [Suggest] check mileage accrual rate
- Operator [Inform] D and I class accrual rate
- Customer [Ask] Super Flyer in New York
- Operator [Inform] Super Flyer status for New York
- Operator [Inform] reservation information
- Operator [Execute] reservation
"""

SUMMARY_USER = "Conversation:\n{convo}"